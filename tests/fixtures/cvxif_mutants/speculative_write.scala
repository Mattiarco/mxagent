package mxagent.cvxif

import chisel3._
import chisel3.util._
import chisel3.experimental.ExtModule

/** MAC MXFP4 verificato (benchmark b08), collegato per nome: il suo Verilog viene
  * dalla run che lo ha generato (o dal riferimento). Porte = contratto di b08.
  */
class MXFP4MacBB extends ExtModule {
  override def desiredName = "MXFP4Mac"
  val acc     = IO(Input(UInt(32.W)))
  val a_scale = IO(Input(UInt(8.W)))
  val a       = IO(Input(UInt(128.W)))
  val b_scale = IO(Input(UInt(8.W)))
  val b       = IO(Input(UInt(128.W)))
  val y       = IO(Output(UInt(32.W)))
}

/** Istruzioni del coprocessore: opcode custom-0, formato R, funct3 = 0.
  *
  * | funct7 | mnemonico       | effetto                                              | rd |
  * |--------|-----------------|------------------------------------------------------|----|
  * | 0      | mx.lda rs1, rs2 | A[8*j .. 8*j+7] = rs1 (8 x E2M1), j = rs2[1:0]       | no |
  * | 1      | mx.ldb rs1, rs2 | idem per B                                           | no |
  * | 2      | mx.scale rs1,rs2| a_scale = rs1[7:0], b_scale = rs2[7:0] (E8M0)        | no |
  * | 3      | mx.mac rd, rs1  | rd = rs1 + Dot(A, B), un solo arrotondamento (FP32)  | sì |
  */
object MxIsa {
  val Opcode = "b0001011".U(7.W)
  val LDA    = 0
  val LDB    = 1
  val SCALE  = 2
  val MAC    = 3
}

/** Coprocessore CORE-V eXtension Interface (CV-X-IF v1.0.0) attorno al MAC MXFP4.
  *
  * Parametri dell'interfaccia: X_NUM_RS = 2, X_ID_WIDTH = idW, X_RFR_WIDTH = X_RFW_WIDTH = 32,
  * X_NUM_HARTS = 1 (X_HARTID_WIDTH = 1), X_DUALREAD = X_DUALWRITE = 0,
  * X_ISSUE_REGISTER_SPLIT = 0 (operandi nella stessa transazione dell'issue). I campi dei
  * pacchetti sono esposti come segnali piatti `<segnale>_<campo>`.
  *
  * Microarchitettura volutamente semplice: una sola istruzione in volo; un'istruzione
  * accettata cambia lo stato architetturale (buffer A/B, scale) solo DOPO il suo commit
  * con commit_kill = 0, quindi un kill non ha nulla da annullare. Ogni istruzione accettata
  * e non uccisa produce esattamente una transazione di result (anche senza write-back);
  * nessun result è speculativo; le istruzioni rifiutate o uccise non producono result.
  */
class MxCvxifCoprocessor(idW: Int) extends Module {
  def this() = this(4) // costruttore senza argomenti per mxagent.Elaborate
  val io = FlatIO(new Bundle {
    // compressed: nessuna istruzione compressa è gestita (accept = 0)
    val compressed_valid       = Input(Bool())
    val compressed_ready       = Output(Bool())
    val compressed_req_instr   = Input(UInt(16.W))
    val compressed_req_hartid  = Input(UInt(1.W))
    val compressed_resp_instr  = Output(UInt(32.W))
    val compressed_resp_accept = Output(Bool())
    // issue
    val issue_valid              = Input(Bool())
    val issue_ready              = Output(Bool())
    val issue_req_instr          = Input(UInt(32.W))
    val issue_req_hartid         = Input(UInt(1.W))
    val issue_req_id             = Input(UInt(idW.W))
    val issue_resp_accept        = Output(Bool())
    val issue_resp_writeback     = Output(Bool())
    val issue_resp_register_read = Output(UInt(2.W))
    // register (con X_ISSUE_REGISTER_SPLIT = 0 viaggia insieme all'issue)
    val register_valid    = Input(Bool())
    val register_ready    = Output(Bool())
    val register_hartid   = Input(UInt(1.W))
    val register_id       = Input(UInt(idW.W))
    val register_rs_0     = Input(UInt(32.W))
    val register_rs_1     = Input(UInt(32.W))
    val register_rs_valid = Input(UInt(2.W))
    // commit (ready implicito)
    val commit_valid       = Input(Bool())
    val commit_hartid      = Input(UInt(1.W))
    val commit_id          = Input(UInt(idW.W))
    val commit_commit_kill = Input(Bool())
    // result
    val result_valid  = Output(Bool())
    val result_ready  = Input(Bool())
    val result_hartid = Output(UInt(1.W))
    val result_id     = Output(UInt(idW.W))
    val result_data   = Output(UInt(32.W))
    val result_rd     = Output(UInt(5.W))
    val result_we     = Output(Bool())
  })

  io.compressed_ready       := true.B
  io.compressed_resp_instr  := 0.U
  io.compressed_resp_accept := false.B

  /** x è "uguale o più recente" di y (id incrementali con wrap-around). */
  private def newerOrEq(x: UInt, y: UInt): Bool = !(x - y)(idW - 1)

  // ------------------------------------------------------------- decodifica
  val instr  = io.issue_req_instr
  val f7     = instr(31, 25)
  val legal  = instr(6, 0) === MxIsa.Opcode && instr(14, 12) === 0.U && f7 <= MxIsa.MAC.U
  val isMac  = f7 === MxIsa.MAC.U
  val needRs = Mux(isMac, "b01".U(2.W), "b11".U(2.W))
  val rsOk   = io.register_valid && (io.register_rs_valid & needRs) === needRs

  // ------------------------------------------------------------- stato
  val sIdle :: sWait :: sResp :: Nil = Enum(3)
  val state = RegInit(sIdle)
  val pId   = Reg(UInt(idW.W))
  val pHart = Reg(UInt(1.W))
  val pOp   = Reg(UInt(2.W))
  val pRd   = Reg(UInt(5.W))
  val pRs0  = Reg(UInt(32.W))
  val pRs1  = Reg(UInt(32.W))
  val pCommitted = Reg(Bool())
  val resData    = Reg(UInt(32.W))

  val aBuf   = RegInit(VecInit(Seq.fill(4)(0.U(32.W))))
  val bBuf   = RegInit(VecInit(Seq.fill(4)(0.U(32.W))))
  val aScale = RegInit(127.U(8.W))
  val bScale = RegInit(127.U(8.W))

  // ------------------------------------------------------------- issue + register
  val idle = state === sIdle
  // le istruzioni non nostre si rifiutano subito; le nostre aspettano gli operandi
  io.issue_ready              := idle && (!legal || rsOk)
  io.register_ready           := io.issue_ready
  io.issue_resp_accept        := legal
  io.issue_resp_writeback     := legal && isMac
  io.issue_resp_register_read := Mux(legal, needRs, 0.U)
  val fire = io.issue_valid && io.issue_ready && legal

  // Commit che riguardano l'istruzione in attesa di issue, arrivati prima del suo handshake
  // (il coprocessore può essere ancora occupato con la precedente, o aspettare gli operandi).
  // Un commit (kill = 0) di un id >= al suo la rende non speculativa; un kill di un id <= al
  // suo la uccide (il kill vale per l'istruzione indicata e per tutte le più recenti).
  val preV    = RegInit(false.B)
  val preId   = Reg(UInt(idW.W))
  val preOk   = Reg(Bool())
  val preKill = Reg(Bool())
  val cOk   = io.commit_valid && !io.commit_commit_kill
  val cKill = io.commit_valid && io.commit_commit_kill
  val curId = io.issue_req_id
  when(!io.issue_valid || io.issue_ready || (preV && preId =/= curId)) {
    preV := false.B
  }
  when(io.issue_valid && !io.issue_ready && io.commit_valid) {
    val sameTxn = preV && preId === curId
    preV    := true.B
    preId   := curId
    preOk   := (sameTxn && preOk) || (cOk && newerOrEq(io.commit_id, curId))
    preKill := (sameTxn && preKill) || (cKill && newerOrEq(curId, io.commit_id))
  }
  val pre = preV && preId === curId
  val committedAtFire = (pre && preOk) || (cOk && newerOrEq(io.commit_id, curId))
  val killedAtFire    = (pre && preKill) || (cKill && newerOrEq(curId, io.commit_id))

  when(fire && f7(1, 0) === MxIsa.LDA.U) { aBuf(io.register_rs_1(1, 0)) := io.register_rs_0 } // MUTANTE
  when(fire) {
    pId   := curId
    pHart := io.issue_req_hartid
    pOp   := f7(1, 0)
    pRd   := instr(11, 7)
    pRs0  := io.register_rs_0
    pRs1  := io.register_rs_1
    pCommitted := committedAtFire
    state := Mux(killedAtFire && !committedAtFire, sIdle, sWait)
  }

  // ------------------------------------------------------------- MAC
  val mac = Module(new MXFP4MacBB)
  mac.acc     := pRs0
  mac.a_scale := aScale
  mac.a       := Cat(aBuf.reverse) // chunk 0 = elementi 0..7 nei bit bassi
  mac.b_scale := bScale
  mac.b       := Cat(bBuf.reverse)

  // ------------------------------------------------------------- esecuzione dopo il commit
  when(state === sWait) {
    val killNow   = cKill && newerOrEq(pId, io.commit_id)
    val commitNow = cOk && newerOrEq(io.commit_id, pId)
    when(pCommitted || (commitNow && !killNow)) {
      switch(pOp) {
        is(MxIsa.LDA.U)   { aBuf(pRs1(1, 0)) := pRs0 }
        is(MxIsa.LDB.U)   { bBuf(pRs1(1, 0)) := pRs0 }
        is(MxIsa.SCALE.U) { aScale := pRs0(7, 0); bScale := pRs1(7, 0) }
      }
      resData := Mux(pOp === MxIsa.MAC.U, mac.y, 0.U)
      state   := sResp
    }.elsewhen(killNow) {
      state := sIdle
    }
  }

  // ------------------------------------------------------------- result
  io.result_valid  := state === sResp
  io.result_hartid := pHart
  io.result_id     := pId
  io.result_data   := resData
  io.result_rd     := pRd
  io.result_we     := pOp === MxIsa.MAC.U
  when(state === sResp && io.result_ready) {
    state := sIdle
  }
}
