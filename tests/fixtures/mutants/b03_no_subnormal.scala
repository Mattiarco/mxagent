import chisel3._
import chisel3.util._

// MUTANTE: dimentica che 2^-127 è subnormale in FP32 (x = 0 -> 0x00000000)
class E8M0ToFP32 extends RawModule {
  val io = FlatIO(new Bundle {
    val x = Input(UInt(8.W))
    val y = Output(UInt(32.W))
  })
  io.y := Mux(io.x === 0xFF.U, "h7FC00000".U(32.W), Cat(0.U(1.W), io.x, 0.U(23.W)))
}
