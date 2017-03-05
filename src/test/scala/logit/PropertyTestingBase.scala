import breeze.linalg._

import org.scalacheck._
import org.scalatest.prop._
import org.scalacheck.Prop._

trait PropertyTestingBase extends Properties with Checkers  {

//  property("minimize coefficients") = {
//    forAll { (minimized: List[DenseVector[Double]], coefficients: Evaluation) =>
//      minimized.foreach { set => set.data.sum > coefficients.coefficients.map( _.estimate).sum }
//    }
//  }
}
