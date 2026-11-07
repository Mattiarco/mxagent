import chisel3._
import chisel3.util._

class MXFP4Dot32 extends RawModule {
  val io = FlatIO(new Bundle {
    val a_scale = Input(UInt(8.W))
    val a       = Input(UInt(128.W))
    val b_scale = Input(UInt(8.W))
    val b       = Input(UInt(128.W))
    val y       = Output(UInt(32.W))
  })
  val prods = (0 until 32).map { i =>
    val m = Module(new E2M1Mul)
    m.io.a := io.a(4 * i + 3, 4 * i)
    m.io.b := io.b(4 * i + 3, 4 * i)
    m.io.p.asUInt
  }
  val tree = Module(new AdderTree)
  tree.io.x := Cat(prods.reverse)
  val pair = Module(new E8M0Pair)
  pair.io.a := io.a_scale
  pair.io.b := io.b_scale
  val conv = Module(new FixedToFP32)
  conv.io.x   := tree.io.y
  conv.io.exp := pair.io.exp
  conv.io.nan := pair.io.nan
  io.y := conv.io.y
}
