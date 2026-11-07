import chisel3._
import chisel3.util._

/** Riferimento b03: E8M0 -> binary32 (2^-127 subnormale, 0xFF -> NaN canonico). */
class E8M0ToFP32 extends RawModule {
  val io = FlatIO(new Bundle {
    val x = Input(UInt(8.W))
    val y = Output(UInt(32.W))
  })
  io.y := MuxCase(Cat(0.U(1.W), io.x, 0.U(23.W)), Seq(
    (io.x === 0xFF.U) -> "h7FC00000".U(32.W),
    (io.x === 0.U)    -> "h00400000".U(32.W)
  ))
}
