import chisel3._
import chisel3.util._

// MUTANTE: connessione UInt -> SInt senza conversione (errore di elaborazione Chisel)
class E2M1Decode extends RawModule {
  val io = FlatIO(new Bundle {
    val a = Input(UInt(4.W))
    val y = Output(SInt(5.W))
  })
  val mag = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(io.a(2, 0))
  io.y := Cat(0.U(1.W), mag)
}
