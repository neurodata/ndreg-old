#ifndef __itkMetamorphosisImageRegistrationMethodv4_hxx
#define __itkMetamorphosisImageRegistrationMethodv4_hxx
#include "itkMetamorphosisImageRegistrationMethodv4.h"

namespace itk
{

template<typename TFixedImage, typename TMovingImage>
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
MetamorphosisImageRegistrationMethodv4()
{
  m_Scale = 1;                        // 1
  m_RegistrationSmoothness = 0.01;    // 0.01
  m_BiasSmoothness = 0.05;            // 0.05
  m_Mu = 10;                          // 10
  m_Sigma = 1;                        // 1
  m_Gamma = 1;                        // 1
  this->SetLearningRate(1e-6);        // 1e-4
  this->SetMinimumLearningRate(1e-12);// 1e-8
  m_MinimumFractionInitialEnergy = 0;  
  m_NumberOfTimeSteps = 4;            // 4 
  m_NumberOfIterations = 20;          // 20
  m_UseJacobian = true;
  m_UseBias = true;
  m_RecalculateEnergy = true;
  this->m_CurrentIteration = 0;
  this->m_IsConverged = false;

  m_VelocityKernel = TimeVaryingImageType::New();                // K_V
  m_InverseVelocityKernel = TimeVaryingImageType::New();         // L_V
  m_RateKernel = TimeVaryingImageType::New();                    // K_R
  m_InverseRateKernel = TimeVaryingImageType::New();             // L_R
  m_Rate = TimeVaryingImageType::New();                          // r
  m_Bias = VirtualImageType::New();                              // B

  m_FixedImageGradientFilter = GradientImageFilter<FixedImageType, double, double>::New();
  m_MovingImageGradientFilter = GradientImageFilter<MovingImageType, double, double>::New();

  typedef typename ImageMetricType::FixedImageGradientImageType::PixelType             FixedGradientPixelType;
  m_FixedImageConstantGradientFilter = FixedImageConstantGradientFilterType::New();
  m_FixedImageConstantGradientFilter->SetConstant(NumericTraits<FixedGradientPixelType>::One);

  typedef typename ImageMetricType::MovingImageGradientImageType::PixelType             MovingGradientPixelType;
  m_MovingImageConstantGradientFilter = MovingImageConstantGradientFilterType::New();
  m_MovingImageConstantGradientFilter->SetConstant(NumericTraits<MovingGradientPixelType>::One);

  this->SetMetric(MeanSquaresImageToImageMetricv4<FixedImageType, MovingImageType>::New());
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::TimeVaryingImagePointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingImagePointer image)
{
  // Calculate the Fourier transform of image.
  typedef ForwardFFTImageFilter<TimeVaryingImageType>    FFTType;
  typename FFTType::Pointer                     fft = FFTType::New();
  fft->SetInput(image);

  //...multiply it by the kernel...
  typedef typename FFTType::OutputImageType  ComplexImageType;
  typedef MultiplyImageFilter<ComplexImageType,TimeVaryingImageType,ComplexImageType>      ComplexImageMultiplierType;
  typename ComplexImageMultiplierType::Pointer  multiplier = ComplexImageMultiplierType::New();
  multiplier->SetInput1(fft->GetOutput());		// Fourier-Transform of image
  multiplier->SetInput2(kernel);			// Kernel

  // ...and finaly take the inverse Fourier transform.
  typedef InverseFFTImageFilter<ComplexImageType,TimeVaryingImageType>  IFFTType;
  typename IFFTType::Pointer                    ifft = IFFTType::New();
  ifft->SetInput(multiplier->GetOutput());
  ifft->Update();

  return ifft->GetOutput();
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::TimeVaryingFieldPointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingFieldPointer field)
{
  // Apply kernel to each component of field
  typedef ComposeImageFilter<TimeVaryingImageType,TimeVaryingFieldType>   ComponentComposerType;
  typename ComponentComposerType::Pointer   componentComposer = ComponentComposerType::New();

  for(unsigned int i = 0; i < ImageDimension; i++)
  {
    typedef VectorIndexSelectionCastImageFilter<TimeVaryingFieldType,TimeVaryingImageType>    ComponentExtractorType;
    typename ComponentExtractorType::Pointer      componentExtractor = ComponentExtractorType::New();
    componentExtractor->SetInput(field);
    componentExtractor->SetIndex(i);
    componentExtractor->Update();

    componentComposer->SetInput(i,ApplyKernel(kernel,componentExtractor->GetOutput()));
  }
  componentComposer->Update();

  return componentComposer->GetOutput();
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
InitializeKernels(TimeVaryingImagePointer kernel, TimeVaryingImagePointer inverseKernel, double alpha, double gamma)
{
  typename TimeVaryingImageType::IndexType   index = this->m_OutputTransform->GetVelocityField()->GetLargestPossibleRegion().GetIndex();
  typename TimeVaryingImageType::SizeType    size = this->m_OutputTransform->GetVelocityField()->GetLargestPossibleRegion().GetSize();
  typename TimeVaryingImageType::SpacingType spacing = this->m_OutputTransform->GetVelocityField()->GetSpacing(); //
  typename TimeVaryingImageType::RegionType region(index,size);

  // Fill in kernels' values
  kernel->CopyInformation(this->m_OutputTransform->GetVelocityField());
  kernel->SetRegions(region);
  kernel->Allocate();

  inverseKernel->CopyInformation(this->m_OutputTransform->GetVelocityField());
  inverseKernel->SetRegions(region);
  inverseKernel->Allocate();

  typedef ImageRegionIteratorWithIndex<TimeVaryingImageType>         TimeVaryingImageIteratorType;
  TimeVaryingImageIteratorType KIt(kernel,kernel->GetLargestPossibleRegion());
  TimeVaryingImageIteratorType LIt(inverseKernel,inverseKernel->GetLargestPossibleRegion());

  for(KIt.GoToBegin(), LIt.GoToBegin(); !KIt.IsAtEnd(); ++KIt,++LIt)
  {
    typename TimeVaryingImageType::IndexType  k = KIt.GetIndex();	// Get the frequency index
    double  A, B;

    // For every dimension accumulate the sum in A
    unsigned int i;
    for(i = 0, A = gamma; i < ImageDimension; i++)
    {
      A += 2 * alpha * vcl_pow(size[i],2) * ( 1.0-cos(2*vnl_math::pi*k[i]/size[i]) );
      //A += 2 * alpha * vcl_pow(spacing[i],-2) * ( 1.0 - cos(2*vnl_math::pi*k[i]*spacing[i]) );
      //A += alpha * vcl_pow(2*vnl_math::pi*k[i]*spacing[i],2);
    }

    KIt.Set(vcl_pow(A,-2)); // Kernel
    LIt.Set(A);             // "Inverse" kernel
  }
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
Initialize()
{
  // Set virtual origin, direction, and starting index based fixed image
  typename FixedImageType::ConstPointer fixedImage = this->GetFixedImage();
  typename VirtualImageType::PointType virtualOrigin = fixedImage->GetOrigin();
  typename VirtualImageType::DirectionType virtualDirection = fixedImage->GetDirection();
  typename VirtualImageType::IndexType virtualIndex = fixedImage->GetLargestPossibleRegion().GetIndex();

  // Set virtual spacing and size based on given scale factor
  typename VirtualImageType::SpacingType virtualSpacing;
  typename VirtualImageType::SizeType virtualSize;
  
  for(unsigned int i = 0; i < ImageDimension; i++)
  {
    virtualSpacing[i] = fixedImage->GetSpacing()[i] / m_Scale;
    virtualSize[i] = vcl_floor(fixedImage->GetLargestPossibleRegion().GetSize(i) * m_Scale) + 1;
  }
  // Adjust virtual size to maximize speed in FFT calculations
  /*
  This filter uses FFT to smooth velocity fields.
  ITK computes FFT using either VNL or FFTW.
  FFTW runs most efficiently when each diminsions size has a small prime factorization.
  This means each diminsion's size has prime factors that are <= 13.
  VnlFFT only works for prime factors 2,3 and 5.
  Therefore we adjust the virtual size to match this critera.
  See "FFT based convolution" by Gaetan Lehmann for more details.
  */
  
  SizeValueType smallPrimes[] = {2,3,5};

  // For each dimension...
  for(unsigned int i = 0; i < virtualSize.GetSizeDimension(); i++)
  {
    // ...while the virtualSize[i] doesn't have a small prime factorization...
    while(true)
    {
      // ...test if virtualSize[i] has a small prime factarization.
      unsigned int n = virtualSize[i];
      for(unsigned int j = 0; j< sizeof(smallPrimes)/sizeof(SizeValueType) && n!=1; j++)
      {
        while(n%smallPrimes[j]==0 && n!=1) { n /= smallPrimes[j];}
      }

      if(n == 1){ break; } // If virtualSize[i]'s has a small prime factorization then we're done.
      virtualSize[i]++;     // Otherwise increment virtualSize[i].
    }
  }

  // Create virtual image
  typename VirtualImageType::RegionType virtualRegion(virtualIndex, virtualSize);

  m_VirtualImage = VirtualImageType::New();
  m_VirtualImage->SetRegions(virtualRegion);
  m_VirtualImage->SetOrigin(virtualOrigin);
  m_VirtualImage->SetSpacing(virtualSpacing);
  m_VirtualImage->SetDirection(virtualDirection);

  // Initialize velocity, v = 0
  typename TimeVaryingFieldType::IndexType velocityIndex;
  velocityIndex.Fill(0);

  typename TimeVaryingFieldType::SizeType  velocitySize;
  velocitySize.Fill(m_NumberOfTimeSteps);

  typename TimeVaryingFieldType::PointType velocityOrigin;
  velocityOrigin.Fill(0);

  typename TimeVaryingFieldType::SpacingType velocitySpacing;
  velocitySpacing.Fill(1);

  typename TimeVaryingFieldType::DirectionType velocityDirection;
  velocityDirection.SetIdentity();    

  for(unsigned int i = 0; i < ImageDimension; i++)
  {
    // Copy information from fixed image
    velocityIndex[i] = virtualIndex[i];
    velocitySize[i] = virtualSize[i];
    velocityOrigin[i] = virtualOrigin[i];
    velocitySpacing[i] = virtualSpacing[i];
    for(unsigned int j = 0; j < ImageDimension; j++)
    {
      velocityDirection(i,j) = virtualDirection(i,j);
    }
  }

  typename TimeVaryingFieldType::RegionType  velocityRegion(velocityIndex,velocitySize);

  TimeVaryingFieldPointer velocity = TimeVaryingFieldType::New();
  velocity->SetRegions(velocityRegion);
  velocity->SetOrigin(velocityOrigin);
  velocity->SetSpacing(velocitySpacing);
  velocity->SetDirection(velocityDirection);
  velocity->Allocate();
  velocity->FillBuffer(NumericTraits<VectorType>::Zero);

  // Initialize rate, r = 0
  m_Rate->SetRegions(velocityRegion);
  m_Rate->CopyInformation(velocity);
  m_Rate->Allocate();
  m_Rate->FillBuffer(NumericTraits<VirtualPixelType>::Zero);

  // Initialize bias, B = 0
  m_Bias->CopyInformation(m_VirtualImage);
  m_Bias->SetRegions(virtualRegion);
  m_Bias->Allocate();
  m_Bias->FillBuffer(NumericTraits<VirtualPixelType>::Zero);

  // Initialize forward image I(1)
  typedef CastImageFilter<MovingImageType, VirtualImageType> CasterType;
  typename CasterType::Pointer caster = CasterType::New();
  caster->SetInput(this->GetMovingImage());
  caster->Update();
  m_ForwardImage = caster->GetOutput();
  
  // Initialize constants
  m_VoxelVolume = 1;
  for(unsigned int i = 0; i < ImageDimension; i++){ m_VoxelVolume *= virtualSpacing[i]; } // \Delta x
  m_NumberOfTimeSteps = velocity->GetLargestPossibleRegion().GetSize()[ImageDimension]; // J
  m_TimeStep = 1.0/(m_NumberOfTimeSteps - 1); // \Delta t

  // Initialize transform
  this->m_OutputTransform->SetVelocityField(velocity);
  this->m_OutputTransform->SetNumberOfIntegrationSteps(m_NumberOfTimeSteps + 2);
  this->m_OutputTransform->SetLowerTimeBound(0.0);
  this->m_OutputTransform->SetUpperTimeBound(1.0);
  this->m_OutputTransform->IntegrateVelocityField();
  
  // Initialize velocity kernels, K_V, L_V
  InitializeKernels(m_VelocityKernel,m_InverseVelocityKernel,m_RegistrationSmoothness,m_Gamma);

  // Initialize rate kernels, K_R, L_R
  InitializeKernels(m_RateKernel,m_InverseRateKernel,m_BiasSmoothness,m_Gamma);
 
  m_RecalculateEnergy = true; // v and r have been initialized
  m_InitialEnergy = GetEnergy();

  this->InvokeEvent(InitializeEvent());
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
CalculateNorm(TimeVaryingImagePointer image)
{
  typedef StatisticsImageFilter<TimeVaryingImageType> CalculatorType;
  typename CalculatorType::Pointer calculator = CalculatorType::New();
  calculator->SetInput(image);
  calculator->Update();
  
  // sumOfSquares = (var(x) + mean(x)^2)*length(x)
  double sumOfSquares = (calculator->GetVariance()+vcl_pow(calculator->GetMean(),2))*image->GetLargestPossibleRegion().GetNumberOfPixels();
  return vcl_sqrt(sumOfSquares*m_VoxelVolume*m_TimeStep);
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
CalculateNorm(TimeVaryingFieldPointer field)
{
  typedef VectorMagnitudeImageFilter<TimeVaryingFieldType,TimeVaryingImageType> MagnitudeFilterType;
  typename MagnitudeFilterType::Pointer magnitudeFilter = MagnitudeFilterType::New();
  magnitudeFilter->SetInput(field);
  magnitudeFilter->Update();

  return CalculateNorm(magnitudeFilter->GetOutput());
}


template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetVelocityEnergy()
{
  return 0.5 * vcl_pow(CalculateNorm(ApplyKernel(m_InverseVelocityKernel,this->m_OutputTransform->GetVelocityField())),2); // 0.5 ||L_V V||^2
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetRateEnergy()
{
  if(m_UseBias)
  {
    return 0.5 * vcl_pow( CalculateNorm(ApplyKernel(m_InverseRateKernel,m_Rate))/m_Mu ,2); // 0.5 \mu^{-2} ||L_R r||^2
  }
  else
  {
    return 0;
  }
}

template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetImageEnergy()
{  
  typedef CastImageFilter<VirtualImageType, MovingImageType> CasterType;
  typename CasterType::Pointer caster = CasterType::New();
  caster->SetInput(m_ForwardImage);                            // I(1)
  caster->Update();
  
  ImageMetricPointer metric = dynamic_cast<ImageMetricType *>(this->m_Metric.GetPointer()); 
  metric->SetFixedImage(this->GetFixedImage());                // I_1
  metric->SetMovingImage(caster->GetOutput());                 // I(1)
  metric->SetFixedImageGradientFilter(m_FixedImageGradientFilter);
  metric->SetMovingImageGradientFilter(m_MovingImageGradientFilter);
  metric->SetVirtualDomainFromImage(m_VirtualImage);
  metric->Initialize();
  
  return 0.5*vcl_pow(m_Sigma,-2) * metric->GetValue() * metric->GetNumberOfValidPoints() * m_VoxelVolume;         // 0.5 \sigma^{-2} ||I(1) - I_1||
}


template<typename TFixedImage, typename TMovingImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetEnergy()
{
  if(m_RecalculateEnergy == true)
  {
    m_Energy = GetVelocityEnergy() + GetRateEnergy() + GetImageEnergy(); // E = E_velocity + E_rate + E_image
    m_RecalculateEnergy = false;
  }
  return m_Energy;
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
IntegrateRate()
{
  typename TimeVaryingImageType::IndexType  index;
  index.Fill(0);

  typename TimeVaryingImageType::SizeType   size = m_Rate->GetLargestPossibleRegion().GetSize();
  size[ImageDimension] = 0;

  typename TimeVaryingImageType::RegionType region(index,size);

  m_Bias->FillBuffer(NumericTraits<VirtualPixelType>::Zero); // B(0) = 0;

  for(unsigned int j = 1; j < m_NumberOfTimeSteps; j++)
  {
    index[ImageDimension] = j-1;
    region.SetIndex(index);

    typedef ExtractImageFilter<TimeVaryingImageType,VirtualImageType> ExtractorType;
    typename ExtractorType::Pointer extractor = ExtractorType::New();
    extractor->SetInput(m_Rate);                    // r
    extractor->SetExtractionRegion(region);
    extractor->SetDirectionCollapseToIdentity();

    typedef MultiplyImageFilter<VirtualImageType,VirtualImageType> MultiplierType;
    typename MultiplierType::Pointer  multiplier = MultiplierType::New();
    multiplier->SetInput(extractor->GetOutput());   // r(j-1)
    multiplier->SetConstant(m_TimeStep);            // \Delta t

    typedef AddImageFilter<VirtualImageType> AdderType;
    typename AdderType::Pointer adder = AdderType::New();
    adder->SetInput1(multiplier->GetOutput());      // r(j-1) \Delta t
    adder->SetInput2(m_Bias);                       // B(j-1)

    this->m_OutputTransform->SetNumberOfIntegrationSteps(2);
    this->m_OutputTransform->SetLowerTimeBound(j * m_TimeStep);     // t_j
    this->m_OutputTransform->SetUpperTimeBound((j-1) * m_TimeStep); // t_{j-1}
    this->m_OutputTransform->IntegrateVelocityField();

    typedef WrapExtrapolateImageFunction<VirtualImageType, RealType>         ExtrapolatorType;
    typedef ResampleImageFilter<VirtualImageType,VirtualImageType,RealType>  ResamplerType;
    typename ResamplerType::Pointer  resampler = ResamplerType::New();
    resampler->SetInput(adder->GetOutput());                    // r(j-1) \Delta t + B(j-1)
    resampler->SetTransform(this->m_OutputTransform);           // \phi_{j,j-1}
    resampler->UseReferenceImageOn();
    resampler->SetReferenceImage(m_VirtualImage);
    resampler->SetExtrapolator(ExtrapolatorType::New());
    resampler->Update();

    m_Bias = resampler->GetOutput();
  }
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::BiasImagePointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetBias()
{
  typedef ResampleImageFilter<VirtualImageType, BiasImageType, RealType>  ResamplerType;
  typename ResamplerType::Pointer resampler = ResamplerType::New();
  resampler->SetInput(m_Bias);   // B(1)
  resampler->UseReferenceImageOn();
  resampler->SetReferenceImage(this->GetFixedImage());
  resampler->Update();

  return resampler->GetOutput();
}

template<typename TFixedImage, typename TMovingImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::FieldPointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GetMetricDerivative(FixedImageGradientFilterPointer fixedImageGradientFilter, MovingImageGradientFilterPointer movingImageGradientFilter)
{
  /* Compute metric derivative p(t) \nabla I(t) */
  typedef DisplacementFieldTransform<RealType,ImageDimension> DisplacementFieldTransformType;
  typename DisplacementFieldTransformType::Pointer backwardTransform = DisplacementFieldTransformType::New();
  backwardTransform->SetDisplacementField(this->m_OutputTransform->GetDisplacementField()); // \phi_{t1}

  typedef CastImageFilter<VirtualImageType, MovingImageType> CasterType;
  typename CasterType::Pointer caster = CasterType::New();
  caster->SetInput(m_ForwardImage); // I(1)
  caster->Update();

  ImageMetricPointer metric = dynamic_cast<ImageMetricType*>(this->m_Metric.GetPointer()); 
  metric->SetFixedImage(this->GetFixedImage());                // I_1
  metric->SetFixedTransform(backwardTransform);                // \phi_{t1}
  metric->SetFixedImageGradientFilter(fixedImageGradientFilter);
  metric->SetMovingImage(caster->GetOutput());                 // I(1)
  metric->SetMovingTransform(backwardTransform);               // \phi_{t1}
  metric->SetMovingImageGradientFilter(movingImageGradientFilter);
  metric->SetVirtualDomainFromImage(m_VirtualImage);
  metric->Initialize();

  // Setup metric derivative
  typename MetricDerivativeType::SizeValueType metricDerivativeSize = m_VirtualImage->GetLargestPossibleRegion().GetNumberOfPixels() * ImageDimension;
  MetricDerivativeType metricDerivative(metricDerivativeSize);
  metricDerivative.Fill(NumericTraits<typename MetricDerivativeType::ValueType>::ZeroValue());

  // Get metric derivative
  metric->GetDerivative(metricDerivative); // -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
  VectorType *metricDerivativePointer = reinterpret_cast<VectorType*> (metricDerivative.data_block());

  SizeValueType numberOfPixelsPerTimeStep = m_VirtualImage->GetLargestPossibleRegion().GetNumberOfPixels();

  typedef ImportImageFilter<VectorType, ImageDimension> ImporterType;
  typename ImporterType::Pointer importer = ImporterType::New();
  importer->SetImportPointer(metricDerivativePointer, numberOfPixelsPerTimeStep, false);
  importer->SetRegion(m_VirtualImage->GetLargestPossibleRegion());
  importer->SetOrigin(m_VirtualImage->GetOrigin());
  importer->SetSpacing(m_VirtualImage->GetSpacing());
  importer->SetDirection(m_VirtualImage->GetDirection());
  importer->Update();

  FieldPointer metricDerivativeField = importer->GetOutput();    

  // ITK dense transforms always return identity for jacobian with respect to parameters.  
  // ... so we provide an option to use it here.
  typedef MultiplyImageFilter<FieldType,VirtualImageType>  FieldMultiplierType;

  if(m_UseJacobian)
  {
    typedef DisplacementFieldJacobianDeterminantFilter<FieldType,RealType,VirtualImageType>  JacobianDeterminantFilterType;
    typename JacobianDeterminantFilterType::Pointer jacobianDeterminantFilter = JacobianDeterminantFilterType::New();
    jacobianDeterminantFilter->SetInput(this->m_OutputTransform->GetDisplacementField()); // \phi_{t1}

    typename FieldMultiplierType::Pointer multiplier0 = FieldMultiplierType::New();
    multiplier0->SetInput1(importer->GetOutput());                  // -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
    multiplier0->SetInput2(jacobianDeterminantFilter->GetOutput()); // |D\phi_{t1}|
    multiplier0->Update();

    metricDerivativeField = multiplier0->GetOutput();
  }

  typename FieldMultiplierType::Pointer multiplier1 = FieldMultiplierType::New();
  multiplier1->SetInput(metricDerivativeField);  // -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
  multiplier1->SetConstant(vcl_pow(m_Sigma,-2)); // 0.5 \sigma^{-2}
  multiplier1->Update();
  
  return multiplier1->GetOutput(); // p(t) \nabla I(t) = -0.5 \sigma^{-2} -dM(I(1) o \phi{t1}, I_1 o \phi{t1})
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
UpdateControls()
{
  typedef JoinSeriesImageFilter<FieldType,TimeVaryingFieldType> FieldJoinerType;
  typename FieldJoinerType::Pointer velocityJoiner = FieldJoinerType::New();

  typedef JoinSeriesImageFilter<VirtualImageType,TimeVaryingImageType> ImageJoinerType;
  typename ImageJoinerType::Pointer rateJoiner = ImageJoinerType::New();

  // For each time step
  for(unsigned int j = 0; j < m_NumberOfTimeSteps; j++)
  {
    double t = j * m_TimeStep;

    // Compute reverse mapping, \phi_{t1} by integrating velocity field, v(t).
    if(j == m_NumberOfTimeSteps-1)
    {
      this->m_OutputTransform->GetModifiableDisplacementField()->FillBuffer(NumericTraits<VectorType>::Zero);
    }
    else
    {
      this->m_OutputTransform->SetNumberOfIntegrationSteps((m_NumberOfTimeSteps-1-j) + 2);
      this->m_OutputTransform->SetLowerTimeBound(t);
      this->m_OutputTransform->SetUpperTimeBound(1.0);
      this->m_OutputTransform->IntegrateVelocityField();
    }
    
    velocityJoiner->PushBackInput(GetMetricDerivative(m_FixedImageGradientFilter,m_MovingImageGradientFilter)); // p(t) \nabla I(t)

    if(m_UseBias)
    {
      typedef VectorIndexSelectionCastImageFilter<FieldType, VirtualImageType> ComponentExtractorType;
      typename ComponentExtractorType::Pointer componentExtractor = ComponentExtractorType::New();
      componentExtractor->SetInput(GetMetricDerivative(dynamic_cast<FixedImageGradientFilterType*>(m_FixedImageConstantGradientFilter.GetPointer()), dynamic_cast<MovingImageGradientFilterType*>(m_MovingImageConstantGradientFilter.GetPointer()))); // p(t) [1,1,1]
      componentExtractor->SetIndex(0);
      componentExtractor->Update();

      rateJoiner->PushBackInput(componentExtractor->GetOutput()); // p(t)
    }

  } // end for j
  velocityJoiner->Update();

  // Compute velocity energy gradient, \nabla_V E = v + K_V [p \nabla I]
  typedef AddImageFilter<TimeVaryingFieldType> TimeVaryingFieldAdderType;
  typename TimeVaryingFieldAdderType::Pointer adder0 = TimeVaryingFieldAdderType::New();
  adder0->SetInput1(this->m_OutputTransform->GetVelocityField());                 // v
  adder0->SetInput2(ApplyKernel(m_VelocityKernel,velocityJoiner->GetOutput()));   // K_V[p \nabla I]
  adder0->Update();
  TimeVaryingFieldPointer velocityEnergyGradient = adder0->GetOutput();                                   // \nabla_V E = v + K_V[p \nabla I]

  // Compute rate energy gradient \nabla_r E = r - \mu^2 K_R[p]
  TimeVaryingImagePointer rateEnergyGradient;
  typedef MultiplyImageFilter<TimeVaryingImageType,TimeVaryingImageType>  TimeVaryingImageMultiplierType;
  typedef AddImageFilter<TimeVaryingImageType>                            TimeVaryingImageAdderType;

  if(m_UseBias)
  {
    rateJoiner->Update();

    typename TimeVaryingImageMultiplierType::Pointer multiplier1 = TimeVaryingImageMultiplierType::New();
    multiplier1->SetInput(ApplyKernel(m_RateKernel,rateJoiner->GetOutput())); // K_R[p]
    multiplier1->SetConstant(-vcl_pow(m_Mu,2));     // -\mu^2

    typename TimeVaryingImageAdderType::Pointer adder1 = TimeVaryingImageAdderType::New();
    adder1->SetInput1(m_Rate);                     // r
    adder1->SetInput2(multiplier1->GetOutput());   // -\mu^2 K_R[p]
    adder1->Update();

    rateEnergyGradient = adder1->GetOutput();      // \nabla_R E = r - \mu^2 K_R[p]
  }

  double                  energyOld = GetEnergy();
  TimeVaryingFieldPointer velocityOld = this->m_OutputTransform->GetVelocityField();
  TimeVaryingImagePointer rateOld = m_Rate;

  while(this->GetLearningRate() > m_MinimumLearningRate && energyOld/m_InitialEnergy > m_MinimumFractionInitialEnergy)
  {
    // Update velocity, v = v - \epsilon \nabla_V E
    typedef MultiplyImageFilter<TimeVaryingFieldType,TimeVaryingImageType>  TimeVaryingFieldMultiplierType;
    typename TimeVaryingFieldMultiplierType::Pointer multiplier2 = TimeVaryingFieldMultiplierType::New();
    multiplier2->SetInput(velocityEnergyGradient);                   // \nabla_V E
    multiplier2->SetConstant(-this->GetLearningRate());              // -\epsilon

    typename TimeVaryingFieldAdderType::Pointer adder2 = TimeVaryingFieldAdderType::New();
    adder2->SetInput1(this->m_OutputTransform->GetVelocityField());   // v
    adder2->SetInput2(multiplier2->GetOutput());                      // -\epsilon \nabla_V E
    adder2->Update();

    this->m_OutputTransform->SetVelocityField(adder2->GetOutput());  // v = v - \epsilon \nabla_V E

    // Compute forward mapping \phi{10} by integrating velocity field v(t)
    this->m_OutputTransform->SetNumberOfIntegrationSteps((m_NumberOfTimeSteps -1) + 2);
    this->m_OutputTransform->SetLowerTimeBound(1.0);
    this->m_OutputTransform->SetUpperTimeBound(0.0);
    this->m_OutputTransform->IntegrateVelocityField();


    typedef DisplacementFieldTransform<RealType,ImageDimension> DisplacementFieldTransformType;
    typename DisplacementFieldTransformType::Pointer transform = DisplacementFieldTransformType::New();
    transform->SetDisplacementField(this->m_OutputTransform->GetDisplacementField()); // \phi_{t1}

    // Compute forward image I(1) = I_0 o \phi_{10} + B(1)
    typedef WrapExtrapolateImageFunction<MovingImageType, RealType>         ExtrapolatorType;
    typedef ResampleImageFilter<MovingImageType,VirtualImageType,RealType>  MovingResamplerType;
    typename MovingResamplerType::Pointer resampler = MovingResamplerType::New();
    resampler->SetInput(this->GetMovingImage());   // I_0
    resampler->SetTransform(transform);     // \phi_{t0}
    resampler->UseReferenceImageOn();
    resampler->SetReferenceImage(this->GetFixedImage());
    resampler->SetExtrapolator(ExtrapolatorType::New());
    resampler->Update();

    m_ForwardImage = resampler->GetOutput();       // I_0 o \phi_{10}

    if(m_UseBias)
    {
      // Update rate, r = r - \epsilon \nabla_R E
      typename TimeVaryingImageMultiplierType::Pointer multiplier3 = TimeVaryingImageMultiplierType::New();
      multiplier3->SetInput(rateEnergyGradient);            // \nabla_R E
      multiplier3->SetConstant(-this->GetLearningRate());   // -\epsilon

      typename TimeVaryingImageAdderType::Pointer adder3 = TimeVaryingImageAdderType::New();
      adder3->SetInput1(m_Rate);                    // r
      adder3->SetInput2(multiplier3->GetOutput());  // -\epsilon \nabla_R E
      adder3->Update();

      m_Rate = adder3->GetOutput(); // r = r - \epsilon \nabla_R E  */
      IntegrateRate();

      typedef AddImageFilter<VirtualImageType>   AdderType;
      typename AdderType::Pointer biasAdder = AdderType::New();
      biasAdder->SetInput1(m_ForwardImage);    // I_0 o \phi_{10}
      biasAdder->SetInput2(GetBias());         // B(1)
      biasAdder->Update();
    
      m_ForwardImage = biasAdder->GetOutput(); // I_0 o \phi_{10} + B(1)
    }

    m_RecalculateEnergy = true;
    
    if(GetEnergy() > energyOld)  // If energy increased...
    {
      // ...restore the controls to their previous values and decrease learning rate
      this->SetLearningRate(0.5*this->GetLearningRate());
      this->m_OutputTransform->SetVelocityField(velocityOld);
      m_Rate = rateOld;
      m_RecalculateEnergy = true;
    }
    else // If energy decreased...
    {
      // ...slightly increase learning rate
      this->SetLearningRate(1.1*this->GetLearningRate());
      return;
    }

  }
  
  m_IsConverged = true;
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
StartOptimization()
{
  this->InvokeEvent(StartEvent());
  for(this->m_CurrentIteration = 0; this->m_CurrentIteration < m_NumberOfIterations; this->m_CurrentIteration++)
  {
    UpdateControls();
    if(this->m_IsConverged){ break; }
    this->InvokeEvent(IterationEvent());
  }
}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
GenerateData()
{
  Initialize();
  StartOptimization();

  // Integrate rate to get final bias, B(1)  
  if(m_UseBias) { IntegrateRate(); }

  /*
  // Pad spatial dimension of velocity
  typedef WrapPadImageFilter<TimeVaryingFieldType, TimeVaryingFieldType> PadderType;
  typename PadderType::SizeType upperBound; upperBound.Fill(1); upperBound[ImageDimension] = 0;
  typename PadderType::Pointer padder = PadderType::New();
  padder->SetInput(this->m_OutputTransform->GetVelocityField());
  padder->SetPadUpperBound(upperBound);
  padder->Update();
  */
  // Integrate velocity to get final displacement, \phi_10
  //this->m_OutputTransform->SetVelocityField(padder->GetOutput());
  this->m_OutputTransform->SetNumberOfIntegrationSteps(m_NumberOfTimeSteps);
  this->m_OutputTransform->SetLowerTimeBound(1.0);
  this->m_OutputTransform->SetUpperTimeBound(0.0);
  this->m_OutputTransform->IntegrateVelocityField();
  this->GetTransformOutput()->Set(this->m_OutputTransform);
  this->InvokeEvent(EndEvent());

}

template<typename TFixedImage, typename TMovingImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage>::
PrintSelf(std::ostream& os, Indent indent ) const
{
  ProcessObject::PrintSelf(os, indent);
  os<<indent<<"Velocity Smoothness: " <<m_RegistrationSmoothness<<std::endl;
  os<<indent<<"Bias Smoothness: "<<m_BiasSmoothness<<std::endl;
}


} // End namespace itk


#endif
