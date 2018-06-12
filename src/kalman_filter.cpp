#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ = F_ * x_; //state transition of x with external motion
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_; //prediction
}

void KalmanFilter::Update_L(const VectorXd &z) {
	// current state and H_
	VectorXd hx_ = H_ * x_;
	//update state and covariance matrix
	UpdateEKF(z, hx_);

}

void KalmanFilter::Update_R(const VectorXd &z) {
	Tools tools;

	//Jacobian using cartesian coordinates
	H_ = MatrixXd(3, 4);
	H_ = tools.CalculateJacobian(tools.PolarToCartesian(z));

	//perform update if px*px + py*py is not close to zero
	if (x_[0] * x_[0] + x_[1] * x_[1] >= .0001)
	{
		VectorXd hx_ = VectorXd(4);
		hx_ = tools.CartesianToPolar(x_);

		//update state and covariance matrix
		UpdateEKF(z, hx_);
	}
}

void KalmanFilter::UpdateEKF(const VectorXd &z, VectorXd &hx_) {
	VectorXd y = z - hx_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//If measurement is from Radar: 
	//Normalize second value phi in the polar coordinate vector y=(rho, phi, rhodot) to be in between -pi and pi
	//using atan2 to return values between -pi and pi
	if (y.size() == 3)
	{
		y(1) = atan2(sin(y(1)), cos(y(1)));
	}

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
