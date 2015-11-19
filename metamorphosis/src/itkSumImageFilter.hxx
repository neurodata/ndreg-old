#ifndef __itkSumImageFilter_hxx
#define __itkSumImageFilter_hxx
#include "itkSumImageFilter.h"

namespace itk
{

template<class TInputImage, class TSum>
void
SumImageFilter<TInputImage, TSum>::
BeforeThreadedGenerateData()
{
  m_Sum = NumericTraits<SumType>::Zero;
  m_ThreadSum.resize(this->GetNumberOfThreads());

  for(unsigned int i = 0; i < this->GetNumberOfThreads(); i++)
  {
    m_ThreadSum[i] = NumericTraits<SumType>::Zero;
  }
}

template< class TInputImage, class TSum>
void
SumImageFilter< TInputImage, TSum >::
ThreadedGenerateData(const RegionType& outputRegionForThread,ThreadIdType threadId)
{
  // Support for progress methods/callbacks
  ProgressReporter  progress(this,threadId,outputRegionForThread.GetNumberOfPixels());
  // Use compensated summation
  CompensatedSummation<SumType>  compensatedSummer;
  ImageRegionConstIterator<TInputImage> it(this->GetInput(),outputRegionForThread);
  while(!it.IsAtEnd())
  {
    compensatedSummer += static_cast<SumType>(it.Get());
    ++it;
    progress.CompletedPixel();
  }
  m_ThreadSum[threadId] = compensatedSummer.GetSum();
}

template<class TInputImage, class TSum>
void
SumImageFilter<TInputImage, TSum>::
AfterThreadedGenerateData()
{
  for(ThreadIdType i = 0; i< this->GetNumberOfThreads(); i++)
  {
    m_Sum += m_ThreadSum[i];
  }
}

} // End namespace itk

#endif
