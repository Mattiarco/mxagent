import chisel3._
import chisel3.util._

// MUTANTE: il subnormale 0.5 viene decodificato come 1.0 (errore tipico: bit implicito sempre a 1)
class E2M1Decode extends RawModule {
  val io = FlatIO(new Bundle {
    val a = Input(UInt(4.W))
    val y = Output(SInt(5.W))
  })
  val mag = VecInit(Seq(0, 2, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(io.a(2, 0))
  val ext = Cat(0.U(1.W), mag)
  io.y := Mux(io.a(3), 0.U(5.W) -% ext, ext).asSInt
}
