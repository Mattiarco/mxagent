import chisel3._
import chisel3.util._

/** Riferimento b05: dot esatto di due blocchi MXFP4, un solo arrotondamento RNE a binary32. */
class MXFP4Dot32 extends RawModule {
  val io = FlatIO(new Bundle {
    val a_scale = Input(UInt(8.W))
    val a       = Input(UInt(128.W))
    val b_scale = Input(UInt(8.W))
    val b       = Input(UInt(128.W))
    val y       = Output(UInt(32.W))
  })

  private def mag(c: UInt): UInt = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(c(2, 0))
  private def tree(xs: Seq[SInt]): SInt =
    if (xs.size == 1) xs.head
    else tree(xs.grouped(2).map { case Seq(l, r) => l +& r; case Seq(l) => l }.toSeq)

  // prodotti esatti in unità di 1/4 (|p| <= 144)
  val prods = (0 until 32).map { i =>
    val x   = io.a(4 * i + 3, 4 * i)
    val z   = io.b(4 * i + 3, 4 * i)
    val ext = Cat(0.U(1.W), mag(x) * mag(z)).asSInt // SInt(9)
    Mux(x(3) ^ z(3), -ext, ext)
  }
  val sum  = tree(prods)                               // |sum| <= 4608
  val neg  = sum < 0.S
  val absS = Mux(neg, -sum, sum).asUInt(12, 0)         // 13 bit
  val nan  = io.a_scale === 0xFF.U || io.b_scale === 0xFF.U
  val zero = absS === 0.U

  // valore = absS * 2^(a_scale + b_scale - 254 - 2); msb = posizione dell'uno più alto
  val msb = Log2(absS)                                  // 0..12
  // esponente polarizzato del risultato normalizzato
  val eb  = (msb.zext +& io.a_scale.zext +& io.b_scale.zext) - 129.S // SInt
  val sig = (absS << (23.U - msb))(23, 0)               // 1.xxx con l'uno al bit 23 (esatto)

  // ramo subnormale: mantissa = round(sig / 2^k), k = 1 - eb >= 1
  val kRaw = 1.S - eb
  val k    = Mux(kRaw > 26.S, 26.U, kRaw.asUInt(4, 0))
  val q    = sig >> k
  val mask = ((1.U(28.W) << k) - 1.U)(26, 0)
  val rem  = Cat(0.U(3.W), sig) & mask
  val half = (1.U(28.W) << k) >> 1
  val up   = rem > half || (rem === half && q(0))
  val sub  = q +& up                                     // 2^23 = minimo normale (riporto corretto)

  val sign = neg.asUInt
  val res = Wire(UInt(32.W))
  when(nan) {
    res := "h7FC00000".U
  }.elsewhen(zero) {
    res := 0.U
  }.elsewhen(eb >= 255.S) {
    res := Cat(sign, "hFF".U(8.W), 0.U(23.W))
  }.elsewhen(eb >= 1.S) {
    res := Cat(sign, eb.asUInt(7, 0), sig(22, 0))
  }.otherwise {
    res := Cat(sign, sub.pad(31)(30, 0))
  }
  io.y := res
}
