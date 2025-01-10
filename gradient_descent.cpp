#include <iostream>
#include <math.h>
#include "gradient_descent.hpp"




// constructor 
gradient_descent::gradient_descent()
{
	// Set defaults.
	nDims = 0;
	priv_learning_rate = 0.0;
	priv_maxIter = 1;
	grad_stepSize = 0.001;
	priv_gradientThresh = 1e-09;
}

gradient_descent::~gradient_descent()
{
	// Tidy up anything that needs tidying up...
}

// function to determine starting vector
void gradient_descent:: set_starting_point(const std::vector<double> start_point){
    priv_start_point = start_point;
    nDims = start_point.size(); 


}

// function to define the object function 
void gradient_descent::set_func(std::function<double(std::vector<double>*)> func){
    my_func = func;
}


// function to set max iter
void gradient_descent:: set_max_iter(int max_iter){
    priv_maxIter = max_iter;
}

void gradient_descent:: set_gradient_thresh(double gradient_thresh){
    priv_gradientThresh = gradient_thresh;
}

void gradient_descent:: set_learning_rate(double learning_rate){
    priv_learning_rate = learning_rate;
}
// method to initialize all variables 
void gradient_descent::init(std::function<double(std::vector<double>*)> func, int max_iter, const std::vector<double> start_point,
                             double gradient_thresh, double learning_rate){
                                set_func(func);
                                set_max_iter(max_iter);
                                set_starting_point(start_point);
                                set_gradient_thresh(gradient_thresh);
                                set_learning_rate(learning_rate);
                                }
// boolean function to perform the optimisation 
bool gradient_descent:: Optimize(std::vector<double> *final_point, double *funcVal) {
    current_point = priv_start_point;
    double gradient_norm = 1.0;
    int iter = 0;
    while (gradient_norm > priv_gradientThresh && iter < priv_maxIter){
        std::vector<double> gradient = Compute_gradient_vector();
        gradient_norm = Compute_gradient_norm(gradient); 
        std::vector<double> new_point = current_point;
        for(int i=0; i<nDims; i++){
            new_point[i] +=  -(gradient[i] * priv_learning_rate);
        }
        current_point = new_point;
        iter++;
    }
    
    *final_point = current_point;
    *funcVal = my_func(&current_point);
    return 0;

}


double gradient_descent :: Compute_gradient(int dim) {
    double gradient = 0.0; 
    std::vector<double> new_point = current_point ; 
    new_point[dim] += grad_stepSize;
    double funcval1 = my_func(&current_point);
    double funcval2 = my_func(&new_point);
    gradient = (funcval2 - funcval1) / grad_stepSize;
    return gradient;
}


// function that returns the gradient vector 
std::vector<double> gradient_descent:: Compute_gradient_vector(){
    std::vector<double> grad_vec = current_point;
    for (int i=0; i<nDims; i++){
        grad_vec[i] = Compute_gradient(i);
    }
    return grad_vec;

}

// function to compute the euclidan norm of the gradient 
double gradient_descent:: Compute_gradient_norm(std:: vector<double> grad_vector){
    double norm = 0.0;
    for (int i=0; i< nDims; i++){
        norm += grad_vector[i] * grad_vector[i];
    }
    return sqrt(norm);
}


