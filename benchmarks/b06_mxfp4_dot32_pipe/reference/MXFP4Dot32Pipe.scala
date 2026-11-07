import chisel3._
import chisel3.util._

/** Riferimento b06: dot MXFP4 di b05 in tre stadi (latenza 3, un ingresso per ciclo).
  * Stadio 1: prodotti e albero di somma. Stadio 2: normalizzazione. Stadio 3: arrotondamento.
  */
class MXFP4Dot32Pipe extends Module {
  val io = FlatIO(new Bundle {
    val a_scale = Input(UInt(8.W))
    val a       = Input(UInt(128.W))
    val b_scale = Input(UInt(8.W))
    val b       = Input(UInt(128.W))
    val y       = Output(UInt(32.W))
  })

  private def mag(c: UInt): UInt = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(c(2, 0))
  // registro con larghezza nota subito (RegNext la lascia all'inferenza: Log2 fallirebbe)
  private def r[T <: Data](x: T): T = { val q = Reg(chiselTypeOf(x)); q := x; q }
  private def tree(xs: Seq[SInt]): SInt =
    if (xs.size == 1) xs.head
    else tree(xs.grouped(2).map { case Seq(l, r) => l +& r; case Seq(l) => l }.toSeq)

  // ---------------------------------------------------------------- stadio 1
  val prods = (0 until 32).map { i =>
    val x   = io.a(4 * i + 3, 4 * i)
    val z   = io.b(4 * i + 3, 4 * i)
    val ext = Cat(0.U(1.W), mag(x) * mag(z)).asSInt
    Mux(x(3) ^ z(3), -ext, ext)
  }
  val sum1    = tree(prods)
  val s1_neg  = r(sum1 < 0.S)
  val s1_abs  = r(Mux(sum1 < 0.S, -sum1, sum1).asUInt(12, 0))
  val s1_nan  = r(io.a_scale === 0xFF.U || io.b_scale === 0xFF.U)
  val s1_esum = r(io.a_scale +& io.b_scale) // 9 bit

  // ---------------------------------------------------------------- stadio 2
  val msb  = Log2(s1_abs)
  val eb   = (msb.zext +& s1_esum.zext) - 129.S
  val sig  = (s1_abs << (23.U - msb))(23, 0)
  val kRaw = 1.S - eb
  val s2_k    = r(Mux(kRaw > 26.S, 26.U, Mux(kRaw < 0.S, 0.U, kRaw.asUInt(4, 0))))
  val s2_sig  = r(sig)
  val s2_eb   = r(eb)
  val s2_neg  = r(s1_neg)
  val s2_nan  = r(s1_nan)
  val s2_zero = r(s1_abs === 0.U)

  // ---------------------------------------------------------------- stadio 3
  val q    = s2_sig >> s2_k
  val mask = ((1.U(28.W) << s2_k) - 1.U)(26, 0)
  val rem  = Cat(0.U(3.W), s2_sig) & mask
  val half = (1.U(28.W) << s2_k) >> 1
  val up   = rem > half || (rem === half && q(0))
  val sub  = q +& up
  val sign = s2_neg.asUInt
  val res  = Wire(UInt(32.W))
  when(s2_nan) {
    res := "h7FC00000".U
  }.elsewhen(s2_zero) {
    res := 0.U
  }.elsewhen(s2_eb >= 255.S) {
    res := Cat(sign, "hFF".U(8.W), 0.U(23.W))
  }.elsewhen(s2_eb >= 1.S) {
    res := Cat(sign, s2_eb.asUInt(7, 0), s2_sig(22, 0))
  }.otherwise {
    res := Cat(sign, sub.pad(31)(30, 0))
  }
  io.y := r(res)
}
