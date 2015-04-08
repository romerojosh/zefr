#ifndef shape_hpp
#define shape_hpp

/* Computes shape function for quadrilateral element */
double calc_shape_quad(unsigned int shape_order, unsigned int idx,
                       double xi, double eta);

/* Computes shape function derivatives for quadrilateral element */
double calc_dshape_quad(unsigned int shape_order, unsigned int idx,
                        double xi, double eta, unsigned int dim);

#endif /* shape_hpp */
