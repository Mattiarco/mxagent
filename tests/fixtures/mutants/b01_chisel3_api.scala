import chisel3._
import chisel3.util._

// MUTANTE: API Chisel 3 obsolete (UInt(4) senza .W, IO con prefisso, Driver)
class E2M1Decode extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(4))
    val y = Output(SInt(5.W))
  })
  io.y := io.a.asSInt
}
