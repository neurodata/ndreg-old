#ifndef __itkMetamorphosisImageRegistrationMethodv4_h
#define __itkMetamorphosisImageRegistrationMethodv4_h

#include "itkVectorIndexSelectionCastImageFilter.h"
#include "itkComposeImageFilter.h"
#include "itkForwardFFTImageFilter.h"
#include "itkInverseFFTImageFilter.h"
#include "itkTimeVaryingVelocityFieldImageRegistrationMethodv4.h"
#include "itkTimeVaryingVelocityFieldTransform.h"
#include "itkVectorCastImageFilter.h"
#include "itkVectorMagnitudeImageFilter.h"
#include "itkGradientImageFilter.h"
#include "itkDisplacementFieldJacobianDeterminantFilter.h"
#include "itkSquareImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkSumImageFilter.h"
#include "itkJoinSeriesImageFilter.h"
#include "itkWrapPadImageFilter.h"
#include "itkExtractImageFilter.h"

namespace itk
{

/** \class MetamorphosisImageRegistrationMethodv4
* \breif Perfoms metamorphosis registration between images
*
* \author Kwane Kutten
*
* \ingroup ITKRegistrationMethodsv4
*/

template<  typename TFixedImage,
           typename TMovingImage = TFixedImage,
           typename TWeightImage = Image<float, TFixedImage::ImageDimension>  >
class MetamorphosisImageRegistrationMethodv4:
public TimeVaryingVelocityFieldImageRegistrationMethodv4<TFixedImage, TMovingImage, TimeVaryingVelocityFieldTransform<float, TFixedImage::ImageDimension> >
{
public:
  /** Standard class typedefs. */
  typedef MetamorphosisImageRegistrationMethodv4                  Self;
  typedef TimeVaryingVelocityFieldImageRegistrationMethodv4<TFixedImage, TMovingImage, TimeVaryingVelocityFieldTransform<float, TFixedImage::ImageDimension> > Superclass;
  typedef SmartPointer<Self>                                      Pointer;
  typedef SmartPointer<const Self>                                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information */
  itkTypeMacro(MetamorphosisImageRegistrationMethodv4, TimeVaryingVelocityFieldImageRegistrationMethodv4);

  /** Concept checking */
#ifdef ITK_USE_CONCEPT_CHECKING
  itkConceptMacro(MovingSameDimensionCheck, (Concept::SameDimension<TFixedImage::ImageDimension,TMovingImage::ImageDimension>));
  itkConceptMacro(WeightSameDimensionCheck, (Concept::SameDimension<TFixedImage::ImageDimension,TWeightImage::ImageDimension>));
  itkConceptMacro(FixedImageHasNumericTraitsCheck, (Concept::HasNumericTraits<typename TFixedImage::PixelType>));
  itkConceptMacro(MovingImageHasNumericTraitsCheck, (Concept::HasNumericTraits<typename TMovingImage::PixelType>));
  itkConceptMacro(WeightImageHasNumericTraitsCheck, (Concept::HasNumericTraits<typename TWeightImage::PixelType>));
#endif

  /** Image typedefs */
  typedef TFixedImage                         FixedImageType;
  typedef typename FixedImageType::Pointer    FixedImagePointer;
  typedef TMovingImage                        MovingImageType;
  typedef typename MovingImageType::Pointer   MovingImagePointer;
  typedef TWeightImage                        WeightImageType;
  typedef typename WeightImageType::Pointer   WeightImagePointer;

  itkStaticConstMacro(ImageDimension, unsigned int, TFixedImage::ImageDimension);

  typedef float                                   PixelType;
  typedef Image<PixelType,ImageDimension>         ImageType;
  typedef typename ImageType::Pointer             ImagePointer;
  typedef Image<PixelType,ImageDimension+1>       TimeVaryingImageType;
  typedef typename TimeVaryingImageType::Pointer  TimeVaryingImagePointer;

  /** Image list types */
  typedef typename std::vector<ImagePointer>    ImageListType;

  typedef typename Superclass::CompositeTransformType                 CompositeTransformType;
  typedef typename Superclass::OutputTransformType                    OutputTransformType;

  typedef typename Superclass::DecoratedOutputTransformType           DecoratedOutputTransformType;
  typedef typename DecoratedOutputTransformType::Pointer              DecoratedOutputTransformPointer;

  typedef typename OutputTransformType::TimeVaryingVelocityFieldType  TimeVaryingFieldType;
  typedef typename TimeVaryingFieldType::Pointer                      TimeVaryingFieldPointer;

  typedef typename OutputTransformType::DisplacementFieldType         FieldType;
  typedef typename FieldType::Pointer                                 FieldPointer;
  typedef typename FieldType::PixelType                               VectorType;
  typedef typename OutputTransformType::ScalarType                    RealType;

  /** Public member functions */
  itkSetMacro(BiasCorrection, bool);
  itkGetConstMacro(BiasCorrection, bool);
  itkBooleanMacro(BiasCorrection);
  itkSetMacro(MinimumLearningRate, double)
  itkGetConstMacro(MinimumLearningRate, double);
  itkSetMacro(RegistrationSmoothness, double);
  itkGetConstMacro(RegistrationSmoothness,double);
  itkSetMacro(BiasSmoothness, double);
  itkGetConstMacro(BiasSmoothness,double);
  itkSetMacro(Sigma,double);
  itkGetConstMacro(Sigma,double);
  itkSetMacro(Mu,double);
  itkGetConstMacro(Mu,double);
  itkSetMacro(NumberOfTimeSteps, unsigned int);
  itkGetConstMacro(NumberOfTimeSteps, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);
  itkGetConstMacro(NumberOfIterations, unsigned int);
  itkGetConstMacro(InitialEnergy, double);
  itkSetMacro(WeightImage, WeightImagePointer);
  itkGetConstMacro(WeightImage, WeightImagePointer);

  double        GetImageEnergy();
  double        GetVelocityEnergy();
  double        GetRateEnergy();
  double        GetEnergy();
  ImagePointer  GetBias() const;

protected:
  MetamorphosisImageRegistrationMethodv4();
  ~MetamorphosisImageRegistrationMethodv4(){};

  TimeVaryingImagePointer ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingImagePointer image);
  TimeVaryingFieldPointer ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingFieldPointer image);
  double CalculateNorm(ImagePointer image);
  double CalculateNorm(TimeVaryingImagePointer image);
  double CalculateNorm(TimeVaryingFieldPointer field);
  void InitializeKernels(TimeVaryingImagePointer kernel, TimeVaryingImagePointer inverseKernel, double alpha, double gamma);
  void Initialize();
  void IntegrateBias();
  void UpdateControls();
  void StartOptimization();
  void GenerateData();
  void PrintSelf(std::ostream& os, Indent indent) const;

private:
  MetamorphosisImageRegistrationMethodv4(const Self&);  // Intentionally not implemened
  void operator=(const Self&);    //Intentionally not implemented

  double        m_Sigma;
  double        m_Mu;
  double        m_Gamma;
  double        m_RegistrationSmoothness;
  double        m_BiasSmoothness;
  double        m_MinimumLearningRate;
  SizeValueType m_NumberOfIterations;
  unsigned int  m_NumberOfTimeSteps;
  double        m_TimeStep;
  double        m_VoxelVolume;
  bool          m_BiasCorrection;
  bool          m_RecalculateEnergy;
  double        m_Energy;
  double        m_InitialEnergy;
  WeightImagePointer      m_WeightImage;
  TimeVaryingImagePointer m_VelocityKernel;
  TimeVaryingImagePointer m_InverseVelocityKernel;
  TimeVaryingImagePointer m_RateKernel;
  TimeVaryingImagePointer m_InverseRateKernel;
  TimeVaryingImagePointer m_Rate;
  ImageListType           m_Bias;

}; // End class MetamorphosisImageRegistrationMethodv4

} // End namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMetamorphosisImageRegistrationMethodv4.hxx"
#endif

#endif
