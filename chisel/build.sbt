// Progetto sbt condiviso da tutti i design (riferimenti dei benchmark e codice generato).
// Versioni BLOCCATE: devono coincidere con mxagent/versions.py (controllato da `mxagent doctor`).
ThisBuild / scalaVersion := "2.13.14"
ThisBuild / version      := "0.1.0"
ThisBuild / organization := "mxagent"

val chiselVersion = "6.6.0"

lazy val root = (project in file("."))
  .settings(
    name := "mxagent-hw",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % chiselVersion
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full),
    scalacOptions ++= Seq(
      "-language:reflectiveCalls",
      "-deprecation",
      "-feature",
      "-Xcheckinit",
      "-Ymacro-annotations"
    ),
    // i sorgenti generati dall'agente vengono messi qui, uno per package
    Compile / unmanagedSourceDirectories += baseDirectory.value / "generated"
  )
