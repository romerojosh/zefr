// GLOBAL SWIG VARIABLES -------------------------------------------------
%typemap(varout) double *dataPy
{
if ($1 == NULL) $result = Py_None;
  else
  {
    PyArrayObject *tmp;
    npy_intp dims[1];
    dims[0] = dataSizePy;
    tmp = (PyArrayObject *)PyArray_SimpleNewFromData(1,dims,PyArray_DOUBLE,(char *)$1);
    $result = (PyObject *)tmp;
  }
}

%typemap(varin) double *dataPy
{
  Py_INCREF($input);
  $1 = ($1_basetype *)(((PyArrayObject *)$input)->data);
}


%typemap(varout) int *idataPy
{
if ($1 == NULL) $result = Py_None;
  else
  {
    PyArrayObject *tmp;
    npy_intp dims[1];
    dims[0] = dataSizePy;
    tmp = (PyArrayObject *)PyArray_SimpleNewFromData(1,dims,PyArray_INT,(char *)$1);
    $result = (PyObject *)tmp;
  }
}

%typemap(varin) int *idataPy
{
  Py_INCREF($input);
  $1 = ($1_basetype *)(((PyArrayObject *)$input)->data);
}
