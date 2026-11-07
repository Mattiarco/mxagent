import chisel3._
import chisel3.util._

/** Riferimento b01: E2M1 -> SInt(5) in unità di 1/2. */
class E2M1Decode extends RawModule {
  val io = FlatIO(new Bundle {
    val a = Input(UInt(4.W))
    val y = Output(SInt(5.W))
  })
  // magnitudine in unità di 1/2 per i codici 0..7: 0, 0.5, 1, 1.5, 2, 3, 4, 6
  val mag = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(io.a(2, 0))
  val ext = Cat(0.U(1.W), mag)                       // 5 bit, positivo
  io.y := Mux(io.a(3), 0.U(5.W) -% ext, ext).asSInt  // -0 -> 0
}
