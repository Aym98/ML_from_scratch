#ifndef GRADIENT_DESCENT
#define GRADIENT_DESCENT
#include <vector>
#include <functional>



class gradient_descent
{
	public:
		// Constructor (default) / destructor.
		gradient_descent();
		~gradient_descent();

        void set_func(std::function<double(std::vector<double>*)> func);

        void set_max_iter(int max_iter);
 
        void set_starting_point(const std::vector<double> start_point);

        void set_gradient_thresh(double gradient_thresh);

        void set_learning_rate(double learning_rate);

        void init(std::function<double(std::vector<double>*)> func, int max_iter, 
                     const std::vector<double> start_point, double gradient_thresh, double learning_rate);

        bool Optimize(std::vector<double> *final_point, double *funcVal);

    private:

        double Compute_gradient(int dim);
        std::vector<double> Compute_gradient_vector();
        double Compute_gradient_norm(std::vector<double> grad_vector);

    //private variables
    private:

    	int nDims;
	    double priv_learning_rate;
	    int priv_maxIter;
	    double grad_stepSize;
	    double priv_gradientThresh;

        std:: vector<double> priv_start_point;
        std:: vector<double> current_point;

        std::function<double(std::vector<double>*)> my_func;
        

};






#endif