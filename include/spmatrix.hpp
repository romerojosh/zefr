#ifndef spmatrix_hpp
#define spmatrix_hpp

#include <algorithm>
#include <vector>

/* spmatrix - Sparse matrix class. 
 * Represents a sparse matrix in CSR format, with construction in COO format.
 * Matrix structure is fixed once conversion to CSR occurs (for now).
 */
template <typename T>
class spmatrix
{
  private:
    std::vector<int> row_ptr; /* Vector for row-pointer (CSR), i index (COO) */
    std::vector<int> col_idx; /* Vector for column index (CSR), j index (COO) */
    std::vector<T> vals; 
    bool isCSR = false;
    int nNonzeros = 0;
    int nRows = 0;

  public:
    /* Method to add entry to COO formatted matrix */
    void addEntry(int i, int j, T val);

    /* Method to get CSR value vector */
    const std::vector<T>& getVals() const; 

    /* Method to get CSR row pointer vector */
    const std::vector<int>& getRowPtr() const; 

    /* Method to get CSR column index vector */
    const std::vector<int>& getColIdx() const; 

    /* Method to get number of rows */
    int getNrows() const; 

    /* Method to get number of nonzero entries */
    int getNnonzeros() const; 

    /* Method to perform in-place conversion of COO formatted matrix to CSR format */ 
    void toCSR();
};

/* Helper function for CSR conversion to sort rows */
template <typename T>
void SortRow(std::vector<int>    &row_idx,
             std::vector<int>    &col_idx, 
             std::vector<T>      &vals,
             int        start,
             int        end,
             int        nz,
             int        current_row)
{
  for (int i = end - 1; i > start; i--)
  {
    for(int j = start; j < i; j++)
    {
      if (col_idx[j] > col_idx[j+1])
      {
        /* Swap the value and the column index */
        double dt = vals[j]; 
        vals[j] = vals[j+1]; 
        vals[j+1] = dt;

        int it = col_idx[j]; 
        col_idx[j] = col_idx[j+1]; 
        col_idx[j+1] = it;
      }
    }
  }

  /* Accumulate duplicate values and adjust vectors */
  for (int j = start; j<end-1; j++)
  {
    if (col_idx[j] == col_idx[j+1])
    {
      vals[j] += vals[j+1];

      for (int i = j+1; i<nz-1; i++)
      {
        vals[i] = vals[i+1];
        col_idx[i] = col_idx[i+1];
      } 

      for (int i = current_row + 1; i<row_idx.size(); i++)
      {
        row_idx[i]--;
      }

      vals[nz-1] = 0;
      end--;
      j--;
    }

  }
}

template <typename T>
void spmatrix<T>::addEntry(int i, int j, T val)
{
  row_ptr.push_back(i);
  col_idx.push_back(j);
  vals.push_back(val);
}

template <typename T>
const std::vector<T>& spmatrix<T>::getVals() const 
{
  return vals;
}

template <typename T>
const std::vector<int>& spmatrix<T>::getRowPtr() const 
{
  return row_ptr;
}


template <typename T>
const std::vector<int>& spmatrix<T>::getColIdx() const 
{
  return col_idx;
}

template <typename T>
int spmatrix<T>::getNrows() const 
{
  return nRows;
}

template <typename T>
int spmatrix<T>::getNnonzeros() const 
{
  return nNonzeros;
}


template <typename T>
void spmatrix<T>::toCSR()
{
  nRows = *std::max_element(row_ptr.begin(), row_ptr.end()) + 1;
  nNonzeros = (int)vals.size();

  /* Handle case with only 1 nonzero per row */
  if (row_ptr.size() < nRows + 1)
    row_ptr.resize(nRows+1);

  std::vector<int> row_start(nRows + 1, 0);

  /* Determine row lengths */
  for (int i = 0; i < nNonzeros; i++) row_start[row_ptr[i]+1]++;
  for (int i = 0; i < nRows; i++) row_start[i+1] += row_start[i];

  for (int init = 0; init < nNonzeros; )
  {
    T dt = vals[init];
    int i = row_ptr[init];
    int j = col_idx[init];
    row_ptr[init] = -1;

    while (true)
    {
      int i_pos = row_start[i];
      T val_next = vals[i_pos];
      int i_next = row_ptr[i_pos];
      int j_next = col_idx[i_pos];

      vals[i_pos] = dt;
      col_idx[i_pos] = j;
      row_ptr[i_pos] = -1;
      row_start[i]++;

      if (i_next < 0) break;

      dt = val_next;
      i = i_next;
      j = j_next;
    }
    init++;

    while ((init < nNonzeros) and (row_ptr[init] < 0))
    {
      init++;
    }
  }

  /* Copy row pointer */
  for (int i = 0; i < nRows; i++)
  {
    row_ptr[i+1] = row_start[i];
  }
  row_ptr[0] = 0;

  /* Sort each row */
  for (int i = 0; i < nRows; i++)
  {
    SortRow(row_ptr, col_idx, vals, row_ptr[i], row_ptr[i+1], nNonzeros, i);
  }

  /* Resize vectors */
  row_ptr.resize(nRows+1);
  vals.resize(nNonzeros);
  col_idx.resize(nNonzeros);
  nNonzeros = row_ptr[nRows];
 
  isCSR = true;
}

#endif /* spmatrix_hpp */
