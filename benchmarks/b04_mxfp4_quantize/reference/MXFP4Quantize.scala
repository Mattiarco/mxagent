import chisel3._
import chisel3.util._

/** Riferimento b04: 32 x binary32 -> blocco MXFP4 (OCP MX §6.3, RNE + saturazione). */
class MXFP4Quantize extends RawModule {
  val io = FlatIO(new Bundle {
    val x     = Input(UInt(1024.W))
    val scale = Output(UInt(8.W))
    val q     = Output(UInt(128.W))
  })

  val xs      = (0 until 32).map(i => io.x(32 * i + 31, 32 * i))
  val signs   = xs.map(_(31))
  val efs     = xs.map(_(30, 23))
  val mfs     = xs.map(_(22, 0))
  val special = xs.map(x => x(30, 23) === 0xFF.U)
  val zero    = xs.map(x => x(30, 0) === 0.U)
  // floor(log2|x|): normali ef-127, subnormali msb(mantissa)-149
  val flog = (0 until 32).map { i =>
    Mux(efs(i) === 0.U, Log2(mfs(i)).zext - 149.S, efs(i).zext - 127.S) // SInt(10)
  }
  // massimo con maschera (solo valori finiti non nulli)
  val cand = (0 until 32).map(i => Mux(zero(i), -512.S(10.W), flog(i)))
  val maxE = VecInit(cand).reduceTree((l, r) => Mux(l > r, l, r))
  val allZero = zero.reduce(_ && _)
  val anySpecial = special.reduce(_ || _)
  val sharedRaw = maxE -& 2.S
  val shared = Mux(allZero || sharedRaw < -127.S, -127.S(11.W), sharedRaw) // <= 125

  // elemento: T = |x| / 2^shared in unità di 1/4 (T < 32), con sticky
  val codes = (0 until 32).map { i =>
    val sig = Cat(efs(i) =/= 0.U, mfs(i))                     // 24 bit
    val ev  = Mux(efs(i) === 0.U, -126.S(10.W), efs(i).zext - 127.S) // esponente del bit 23
    val kS  = shared +& 21.S -& ev                            // >= 19
    val k   = Mux(kS > 40.S, 40.U, kS.asUInt(5, 0))
    val t   = (sig >> k)(4, 0)
    val mask = ((1.U(41.W) << k) - 1.U)(23, 0)
    val sticky = (sig & mask) =/= 0.U
    // regioni: [0,2) passo 1/2, [2,4) passo 1, [4,8) passo 2 (unità di 1/4: 2, 4, 8)
    val code = Wire(UInt(3.W))
    when(t < 8.U) {
      val q2 = t >> 1
      val up = t(0) && (sticky || q2(0))
      code := q2 +& up                                         // 0..4 -> codici 0..4
    }.elsewhen(t < 16.U) {
      val q1 = t >> 2                                          // 2 o 3
      val up = t(1) && (t(0) || sticky || q1(0))
      code := (q1 +& up) + 2.U                                 // 2->4, 3->5, 4->6
    }.otherwise {
      val q0 = t >> 3                                          // 2 o 3 (4.0 o 6.0)
      val up = t(2) && (t(1, 0) =/= 0.U || sticky || q0(0))
      code := Mux(q0 === 3.U || up, 7.U, 6.U)                  // 8.0 satura a 6.0
    }
    Cat(signs(i), code)
  }
  io.scale := Mux(anySpecial, 0xFF.U, (shared + 127.S).asUInt(7, 0))
  io.q     := Mux(anySpecial, 0.U, Cat(codes.reverse))
}
