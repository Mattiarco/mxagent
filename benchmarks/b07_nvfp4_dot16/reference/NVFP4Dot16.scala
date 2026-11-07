import chisel3._
import chisel3.util._

/** Riferimento b07: dot NVFP4 (16 x E2M1, scale E4M3). Il risultato è sempre esatto in binary32. */
class NVFP4Dot16 extends RawModule {
  val io = FlatIO(new Bundle {
    val a_scale = Input(UInt(8.W))
    val a       = Input(UInt(64.W))
    val b_scale = Input(UInt(8.W))
    val b       = Input(UInt(64.W))
    val y       = Output(UInt(32.W))
  })

  private def mag(c: UInt): UInt = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(c(2, 0))
  private def tree(xs: Seq[SInt]): SInt =
    if (xs.size == 1) xs.head
    else tree(xs.grouped(2).map { case Seq(l, r) => l +& r; case Seq(l) => l }.toSeq)

  // E4M3: valore = sig * 2^(E - 10), sig = 1.mmm (normale) o 0.mmm (subnormale, E = 1)
  private def sigOf(c: UInt): UInt = Cat(c(6, 3) =/= 0.U, c(2, 0)) // 4 bit
  private def expOf(c: UInt): UInt = Mux(c(6, 3) === 0.U, 1.U(4.W), c(6, 3))

  val prods = (0 until 16).map { i =>
    val x   = io.a(4 * i + 3, 4 * i)
    val z   = io.b(4 * i + 3, 4 * i)
    val ext = Cat(0.U(1.W), mag(x) * mag(z)).asSInt
    Mux(x(3) ^ z(3), -ext, ext)
  }
  val sum  = tree(prods)                                  // unità 1/4, |sum| <= 2304
  val neg  = sum < 0.S
  val absS = Mux(neg, -sum, sum).asUInt(11, 0)
  val p    = absS * sigOf(io.a_scale) * sigOf(io.b_scale) // < 2^20, esatto
  val nan  = io.a_scale(6, 0) === 0x7F.U || io.b_scale(6, 0) === 0x7F.U
  val zero = p === 0.U

  // valore = p * 2^(Ea + Eb - 22): esponente polarizzato = msb + Ea + Eb + 105 (sempre normale)
  val msb  = Log2(p)
  val eb   = msb +& expOf(io.a_scale) +& expOf(io.b_scale) +& 105.U
  val sig  = (p << (23.U - msb))(22, 0)
  val sign = neg ^ io.a_scale(7) ^ io.b_scale(7)

  io.y := Mux(nan, "h7FC00000".U, Mux(zero, 0.U, Cat(sign, eb(7, 0), sig)))
}
