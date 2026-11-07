import chisel3._
import chisel3.util._

// MUTANTE: loop combinatorio (rilevato da firtool)
class E2M1Decode extends RawModule {
  val io = FlatIO(new Bundle {
    val a = Input(UInt(4.W))
    val y = Output(SInt(5.W))
  })
  val t = Wire(UInt(5.W))
  t := t + io.a
  io.y := t.asSInt
}
