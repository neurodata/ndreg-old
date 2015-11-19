#ifndef __itkSumImageFilter_h
#define __itkSumImageFilter_h

#include <vector>
/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "itkNumericTraits.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageToImageFilter.h"
#include "itkCompensatedSummation.h"

namespace itk
{
/** \class StatisticsImageFilter
 * \brief Compute sum of an Image.
 *
 * SumImageFilter computes the  sum of an image.
 * The filter passes its input through unmodified.  The filter is
 * threaded. It computes sum for each thread then combines them in
 * its AfterThreadedGenerate method.
 *
 * \ingroup MathematicalStatisticsImageFilters
 * \ingroup ITKImageStatistics
 * \ingroup MultiThreaded
 *
 */
template<class TInputImage, class TSum = typename NumericTraits<typename TInputImage::PixelType>::RealType  >
class ITK_EXPORT SumImageFilter: public ImageToImageFilter<TInputImage,TInputImage>
{
public:
  /** Standard typedefs*/
  typedef SumImageFilter                                Self;
  typedef ImageToImageFilter<TInputImage,TInputImage>   Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Image Typedefs */
  typedef typename TInputImage::RegionType  RegionType;
  typedef typename TInputImage::PixelType   PixelType;
  typedef TSum                              SumType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(SumImageFilter,ImageToImageFilter);

  /** Public member functions */
  itkGetConstMacro(Sum,SumType);

protected:
  SumImageFilter(){}
  ~SumImageFilter(){}

  void BeforeThreadedGenerateData();
  void ThreadedGenerateData(const RegionType& outputRegionForThread,ThreadIdType threadId);
  void AfterThreadedGenerateData();

private:
  SumImageFilter(const Self&);  // Purposely not implemented
  void operator=(const Self&);  // Purposely not implemented

  SumType              m_Sum;
  std::vector<SumType> m_ThreadSum;
};

} // End namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSumImageFilter.hxx"
#endif

#endif
