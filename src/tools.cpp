#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if (estimations.size() != ground_truth.size()
		|| estimations.size() == 0) {
		cout << "Invalid estimation or ground_truth data" << endl;
		return rmse;
	}

	//accumulate squared residuals
	for (unsigned int i = 0; i < estimations.size(); ++i) {

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3, 4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float term1 = px * px + py * py;
	float term2 = sqrt(term1);
	float term3 = (term1*term2);

	//check division by zero
	if (fabs(term1) < 0.0001) {
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px / term2), (py / term2), 0, 0,
		-(py / term1), (px / term1), 0, 0,
		py*(vx*py - vy * px) / term3, px*(px*vy - py * vx) / term3, px / term2, py / term2;

	return Hj;
}


/**

* Convert polar to cartesian, derive based on the equations

rho = sqrt(px * px + py * py), phi = atan(py/px), rhodot = (px*vx + py*vy) / sqrt(px*px + py*py)

*/

VectorXd Tools::PolarToCartesian(const VectorXd& polar) {

	VectorXd cartesian = VectorXd(4);
	float rho = polar(0);
	float phi = polar(1);
	float rhodot = polar(2);

	float px = rho * cos(phi);
	float py = rho * sin(phi);
	float vx = rhodot * cos(phi);
	float vy = rhodot * sin(phi);


	cartesian << px, py, vx, vy;
	return cartesian;
}



/**

* Convert cartesian to polar coordinates

*/
VectorXd Tools::CartesianToPolar(const VectorXd& cartesian) {

	VectorXd polar = VectorXd(3);
	float thrsh_ = 0.0001;
	float px = cartesian(0);
	float py = cartesian(1);
	float vx = cartesian(2);
	float vy = cartesian(3);

	float rho = sqrt(px*px + py * py);
	float phi = 0;
	float rhodot = 0;



	//handle undefined value for phi, if px and py are zero
	if (px != 0 && py != 0) {
		phi = atan2(py, px);
	}
	//Avoid Divide by Zero error, if radian distance from origin is small
	if (rho > thrsh_) {
		rhodot = (px*vx + py * vy) / rho;
	}

	polar << rho, phi, rhodot;
	return polar;
}