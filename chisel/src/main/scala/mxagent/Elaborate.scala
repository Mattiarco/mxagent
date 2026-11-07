package mxagent

import chisel3._
import circt.stage.ChiselStage

/** Modulo minimo usato da `mxagent doctor --warmup`: scarica Chisel, plugin e
  * firtool nelle cache e verifica la catena sbt -> Chisel -> firtool -> SystemVerilog.
  */
class Smoke extends Module {
  val io = FlatIO(new Bundle {
    val a = Input(UInt(4.W))
    val y = Output(UInt(4.W))
  })
  io.y := RegNext(io.a, 0.U)
}

/** Elaborazione generica: `runMain mxagent.Elaborate <classe> <target-dir>`.
  * La classe deve avere un costruttore senza argomenti.
  */
object Elaborate {
  // niente -strip-debug-info: i locatori `// @[File.scala l:c]` nel Verilog
  // permettono di riportare i warning di lint al sorgente Scala
  // --mlir-disable-threading: niente thread pool di MLIR (un thread per core).
  // Con limiti su processi/thread (ulimit -u, cgroup pids) e job in parallelo,
  // la creazione dei thread fallisce con EAGAIN e firtool abortisce (exit 134)
  // dopo aver già scritto il Verilog. I moduli sono piccoli: il costo è nullo.
  val firtoolOpts = Array("-disable-all-randomization", "--mlir-disable-threading")

  def main(args: Array[String]): Unit = {
    val cls = args.lift(0).getOrElse("mxagent.Smoke")
    val dir = args.lift(1).getOrElse("build/" + cls.split('.').last)
    val gen = () => Class.forName(cls).getDeclaredConstructor().newInstance().asInstanceOf[RawModule]
    ChiselStage.emitSystemVerilogFile(gen(), Array("--target-dir", dir), firtoolOpts)
  }
}
