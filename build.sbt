name := "breeze-demo"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "com.github.scopt" %% "scopt" % "3.5.0",

  "org.apache.spark" %% "spark-core" % "2.0.2" % "provided",
  "org.apache.spark" %% "spark-mllib" % "2.0.2" % "provided",

  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12",
  "org.scalanlp" %% "breeze-viz" % "0.12",

  "org.scalatest" %% "scalatest" % "3.0.0"
)

resolvers += Resolver.sonatypeRepo("public")

parallelExecution in Test := false
