
name := "IrisSpark"

version := "1.0"
scalaVersion := "2.10.5"
resolvers += "scalaz-bintray" at "http://dl.bintray.com/scalaz/releases"
//libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.6.0-cdh5.10.0"
//libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.6.0-cdh5.10.0"
libraryDependencies += "org.apache.spark" % "spark-core_2.10" % "1.6.3"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.10" % "1.6.3"
packAutoSettings
    