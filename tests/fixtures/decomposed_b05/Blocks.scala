import chisel3._
import chisel3.util._

// b05 decomposto a mano con i blocchi del catalogo (collaudo del catalogo, non visibile all'agente)

class E2M1Mul extends RawModule {
  val io = FlatIO(new Bundle {
    val a = Input(UInt(4.W))
    val b = Input(UInt(4.W))
    val p = Output(SInt(9.W))
  })
  private def mag(c: UInt): UInt = VecInit(Seq(0, 1, 2, 3, 4, 6, 8, 12).map(_.U(4.W)))(c(2, 0))
  val ext = Cat(0.U(1.W), mag(io.a) * mag(io.b))
  io.p := Mux(io.a(3) ^ io.b(3), 0.U(9.W) -% ext, ext).asSInt
}

class AdderTree extends RawModule {
  val io = FlatIO(new Bundle {
    val x = Input(UInt(288.W))
    val y = Output(SInt(14.W))
  })
  private def tree(xs: Seq[SInt]): SInt =
    if (xs.size == 1) xs.head
    else tree(xs.grouped(2).map { case Seq(l, r) => l +& r; case Seq(l) => l }.toSeq)
  io.y := tree((0 until 32).map(i => io.x(9 * i + 8, 9 * i).asSInt))
}

class E8M0Pair extends RawModule {
  val io = FlatIO(new Bundle {
    val a   = Input(UInt(8.W))
    val b   = Input(UInt(8.W))
    val exp = Output(SInt(10.W))
    val nan = Output(Bool())
  })
  io.nan := io.a === 0xFF.U || io.b === 0xFF.U
  io.exp := Mux(io.nan, 0.S, (io.a.zext +& io.b.zext) - 254.S)
}

class FixedToFP32 extends RawModule {
  val io = FlatIO(new Bundle {
    val x   = Input(SInt(14.W))
    val exp = Input(SInt(10.W))
    val nan = Input(Bool())
    val y   = Output(UInt(32.W))
  })
  val neg  = io.x < 0.S
  val absX = Mux(neg, -io.x, io.x).asUInt(13, 0)
  val msb  = Log2(absX)
  val eb   = (msb.zext +& io.exp) + 125.S          // x * 2^(exp-2): bias 127 - frac 2
  val sig  = (absX << (23.U - msb))(23, 0)
  val kRaw = 1.S - eb
  val k    = Mux(kRaw > 26.S, 26.U, kRaw.asUInt(4, 0))
  val q    = sig >> k
  val mask = ((1.U(28.W) << k) - 1.U)(26, 0)
  val rem  = Cat(0.U(3.W), sig) & mask
  val half = (1.U(28.W) << k) >> 1
  val up   = rem > half || (rem === half && q(0))
  val sub  = q +& up
  val sign = neg.asUInt
  io.y := MuxCase(Cat(sign, sub.pad(31)(30, 0)), Seq(
    io.nan            -> "h7FC00000".U(32.W),
    (absX === 0.U)    -> 0.U(32.W),
    (eb >= 255.S)     -> Cat(sign, "hFF".U(8.W), 0.U(23.W)),
    (eb >= 1.S)       -> Cat(sign, eb.asUInt(7, 0), sig(22, 0))
  ))
}
