import chisel3._
import chisel3.util._

// MUTANTE: segno del prodotto calcolato con OR invece di XOR
class E2M1Mul extends RawModule {
  val io = FlatIO(new Bundle {
    val a = Input(UInt(4.W))
    val b = Input(UInt(4.W))
    val p = Output(SInt(9.W))
  })
  private def mag(c: UInt): UInt = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(c(2, 0))
  val prod = mag(io.a) * mag(io.b)
  val ext  = Cat(0.U(1.W), prod)
  io.p := Mux(io.a(3) | io.b(3), 0.U(9.W) -% ext, ext).asSInt
}
