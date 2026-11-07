import chisel3._
import chisel3.util._

/** Riferimento b08: y = acc + Dot(A, B) con un solo arrotondamento RNE a binary32.
  *
  * Somma esatta in virgola fissa, unità 2^-152 (W = 282 bit, fino a 2^130). I bit del dot
  * sotto 2^-152 vanno in uno sticky con semantica floor (valore vero in (S, S+1)); il punto
  * T = 2S + sticky (unità 2^-153) cade strettamente dentro l'intervallo e non è mai un punto
  * di pareggio (i pareggi sono multipli di 8), quindi arrotondare T equivale ad arrotondare
  * il valore esatto. Un dot >= 2^129 dà comunque overflow (|acc| < 2^128).
  */
class MXFP4Mac extends RawModule {
  val io = FlatIO(new Bundle {
    val acc     = Input(UInt(32.W))
    val a_scale = Input(UInt(8.W))
    val a       = Input(UInt(128.W))
    val b_scale = Input(UInt(8.W))
    val b       = Input(UInt(128.W))
    val y       = Output(UInt(32.W))
  })
  val W = 282

  private def mag(c: UInt): UInt = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(c(2, 0))
  private def tree(xs: Seq[SInt]): SInt =
    if (xs.size == 1) xs.head
    else tree(xs.grouped(2).map { case Seq(l, r) => l +& r; case Seq(l) => l }.toSeq)

  // ------------------------------------------------------------- dot esatto
  val prods = (0 until 32).map { i =>
    val x   = io.a(4 * i + 3, 4 * i)
    val z   = io.b(4 * i + 3, 4 * i)
    val ext = Cat(0.U(1.W), mag(x) * mag(z)).asSInt
    Mux(x(3) ^ z(3), -ext, ext)
  }
  val sum      = tree(prods) // unità 1/4
  val dotNeg   = sum < 0.S
  val absS     = Mux(dotNeg, -sum, sum).asUInt(12, 0)
  val dotZero  = absS === 0.U
  val scaleNaN = io.a_scale === 0xFF.U || io.b_scale === 0xFF.U

  // dot = absS * 2^(sa + sb - 256) = absS * 2^(sa + sb - 104) unità di finestra
  val sh      = (io.a_scale.zext +& io.b_scale.zext) - 104.S // -104 .. 404
  val dotHuge = !dotZero && (sh +& Log2(absS).zext >= (W - 1).S)
  val left    = Mux(sh >= 0.S, sh.asUInt, 0.U)(8, 0)
  val right   = Mux(sh < 0.S, (0.S - sh).asUInt, 0.U)(6, 0)
  val magD    = Mux(sh >= 0.S, (absS << left)(W - 1, 0), (absS >> right).pad(W))
  val lowMask = ((1.U(128.W) << right) - 1.U)(12, 0)
  val sticky  = sh < 0.S && (absS & lowMask) =/= 0.U
  // floor: un dot negativo con resto diventa -(magD + 1)
  val dFix = Mux(dotNeg, 0.S - (magD +& sticky.asUInt).zext, magD.zext)

  // ------------------------------------------------------------ accumulatore
  val accS   = io.acc(31)
  val accE   = io.acc(30, 23)
  val accM   = io.acc(22, 0)
  val accNaN = accE === 255.U && accM =/= 0.U
  val accInf = accE === 255.U && accM === 0.U
  val sigAcc = Cat(accE =/= 0.U, accM)                          // 24 bit
  val shA    = Mux(accE === 0.U, 1.U(9.W), accE.pad(9)) +& 2.U  // acc = sig * 2^(E-150)
  val magA   = (sigAcc << shA)(W - 1, 0)
  val aFix   = Mux(accS.asBool, 0.S - magA.zext, magA.zext)

  // --------------------------------------------------- somma e arrotondamento
  val s    = aFix +& dFix
  val t    = (s << 1) +& sticky.asUInt.zext // unità 2^-153
  val tNeg = t < 0.S
  val mT   = Mux(tNeg, 0.S - t, t).asUInt
  val zero = mT === 0.U
  val m    = Log2(mT)
  val shR  = Mux(m >= 27.U, m - 23.U, 4.U) // 4 = posizione di 2^-149 (subnormali)
  val q    = mT >> shR
  val one  = 1.U(1.W) << shR
  val rem  = mT & (one - 1.U)
  val half = one >> 1
  val up   = rem > half || (rem === half && q(0))
  val qr   = q +& up.asUInt
  val enc  = ((shR - 4.U) << 23) +& qr // il riporto dell'arrotondamento entra nell'esponente
  val ovf  = enc >= "h7F800000".U

  val res = Wire(UInt(32.W))
  when(scaleNaN || accNaN) {
    res := "h7FC00000".U
  }.elsewhen(accInf) {
    res := io.acc
  }.elsewhen(dotHuge) {
    res := Cat(dotNeg, "hFF".U(8.W), 0.U(23.W))
  }.elsewhen(zero) {
    res := 0.U
  }.elsewhen(ovf) {
    res := Cat(tNeg, "hFF".U(8.W), 0.U(23.W))
  }.otherwise {
    res := Cat(tNeg, enc(30, 0))
  }
  io.y := res
}
