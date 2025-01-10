#include <iostream>
#include <functional>
#include "gradient_descent.hpp"



double simple_func(std::vector<double> *my_point){
    double x = my_point->at(0);
    return x * x;
}

int main(){
    std::function<double(std::vector<double>*)> p_myfunc = {simple_func};
	double learning_rate = 0.1;
	int maxIter = 100;
	double grad_stepSize = 0.0001;
	double gradientThresh = 0.0001;
    std:: vector<double> start_point = {2.5};

    gradient_descent optim;
    optim.init(p_myfunc, maxIter, start_point, gradientThresh, learning_rate);
    std::vector<double> funcLoc;
	double funcVal;

    optim.Optimize(&funcLoc, &funcVal);

    std::cout << "The optimal point is : " << funcLoc[0] << std::endl;
    std::cout <<"The optimal function value is: " <<funcVal<< std::endl; 

    return 0;


}