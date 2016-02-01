#ifndef __itkMetamorphosisImageRegistrationMethodv4_hxx
#define __itkMetamorphosisImageRegistrationMethodv4_hxx

namespace itk
{
template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
MetamorphosisImageRegistrationMethodv4()
{
  this->m_CurrentIteration = 0;
  this->m_IsConverged = false;

  m_Sigma = 1;                        // 1
  m_Mu = 10;                          // 10
  m_Gamma = 1;                        // 1
  m_RegistrationSmoothness = 0.01;    // 0.01
  m_BiasSmoothness = 0.05;            // 0.05
  m_NumberOfTimeSteps = 4;            // 4
  m_NumberOfIterations = 100;         // 20
  m_BiasCorrection = true;            // true
  this->SetLearningRate(1e-1);        // 1e-1
  this->SetMinimumLearningRate(1e-6); // 1e-4
  m_WeightImage = NULL;                                          // W
  m_VelocityKernel = TimeVaryingImageType::New();                // K_V
  m_InverseVelocityKernel = TimeVaryingImageType::New();         // L_V
  m_RateKernel = TimeVaryingImageType::New();                    // K_H
  m_InverseRateKernel = TimeVaryingImageType::New();             // L_H
  m_Rate = TimeVaryingImageType::New();                          // \eta
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::TimeVaryingImagePointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
ApplyKernel(TimeVaryingImagePointer kernel, TimeVaryingImagePointer image)
{
  // Pad image to size of kernel.
  typedef WrapPadImageFilter<TimeVaryingImageType,TimeVaryingImageType>   PadderType;
  typename PadderType::Pointer padder = PadderType::New();
  padder->SetInput(image);
  padder->SetPadUpperBound( kernel->GetLargestPossibleRegion().GetSize() - image->GetLargestPossibleRegion().GetSize() );

  // Calculate the Fourier transform of image.
  typedef ForwardFFTImageFilter<TimeVaryingImageType>    FFTType;
  typename FFTType::Pointer                     fft = FFTType::New();
  fft->SetInput(padder->GetOutput());

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

  // Crop result to original size.
  typedef ExtractImageFilter<TimeVaryingImageType,TimeVaryingImageType> ExtractorType;
  typename ExtractorType::Pointer extractor = ExtractorType::New();
  extractor->SetInput(ifft->GetOutput());
  extractor->SetExtractionRegion(image->GetLargestPossibleRegion());
  extractor->Update();

  return extractor->GetOutput();
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::TimeVaryingFieldPointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
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

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
CalculateNorm(ImagePointer image)
{
  typedef SquareImageFilter<ImageType,ImageType> SquarerType;
  typename SquarerType::Pointer squarer = SquarerType::New();
  squarer->SetInput(image);

  typedef SumImageFilter<ImageType>   SummerType;
  typename SummerType::Pointer summer = SummerType::New();
  summer->SetInput(squarer->GetOutput());
  summer->Update();
  
  return vcl_sqrt(summer->GetSum()*m_VoxelVolume);
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
CalculateNorm(TimeVaryingImagePointer image)
{
  typedef SquareImageFilter<TimeVaryingImageType,TimeVaryingImageType> SquarerType;
  typename SquarerType::Pointer squarer = SquarerType::New();
  squarer->SetInput(image);

  typedef SumImageFilter<TimeVaryingImageType>   SummerType;
  typename SummerType::Pointer summer = SummerType::New();
  summer->SetInput(squarer->GetOutput());
  summer->Update();

  return vcl_sqrt(summer->GetSum()*m_VoxelVolume*m_TimeStep);
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
CalculateNorm(TimeVaryingFieldPointer field)
{
  typedef VectorMagnitudeImageFilter<TimeVaryingFieldType,TimeVaryingImageType> MagnitudeFilterType;
  typename MagnitudeFilterType::Pointer magnitudeFilter = MagnitudeFilterType::New();
  magnitudeFilter->SetInput(field);
  magnitudeFilter->Update();

  return CalculateNorm(magnitudeFilter->GetOutput());
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
GetImageEnergy()
{
  this->m_OutputTransform->SetNumberOfIntegrationSteps(m_NumberOfTimeSteps + 2);
  this->m_OutputTransform->SetLowerTimeBound(1);
  this->m_OutputTransform->SetUpperTimeBound(0);
  this->m_OutputTransform->IntegrateVelocityField();

  typedef ResampleImageFilter<MovingImageType,ImageType,RealType>  MovingResamplerType;
  typename MovingResamplerType::Pointer resampler = MovingResamplerType::New();
  resampler->SetInput(this->GetMovingImage());         // I_0
  resampler->SetTransform(this->m_OutputTransform);    // \phi_{10}
  resampler->UseReferenceImageOn();
  resampler->SetReferenceImage(this->GetFixedImage());

  typedef AddImageFilter<ImageType>   AdderType;
  typename AdderType::Pointer  adder = AdderType::New();
  adder->SetInput1(resampler->GetOutput());           // I_0 o \phi_{10}
  adder->SetInput2(GetBias());                        // B(1)

  typedef SubtractImageFilter<ImageType,FixedImageType,ImageType>  SubtractorType;
  typename SubtractorType::Pointer subtractor = SubtractorType::New();
  subtractor->SetInput1(adder->GetOutput());          // I(1) = I_0 o \phi_{10} + B(1)
  subtractor->SetInput2(this->GetFixedImage());       // I_1
  subtractor->Update();
  
  typename ImageType::Pointer difference = subtractor->GetOutput(); // I(1) - I_1

  if(this->GetWeightImage().GetPointer() != NULL)
  {
    typedef ResampleImageFilter<WeightImageType,ImageType,RealType>  WeightResamplerType;
    typename WeightResamplerType::Pointer weightResampler = WeightResamplerType::New();
    weightResampler->SetInput(this->GetWeightImage());         // W_0
    weightResampler->SetTransform(this->m_OutputTransform);    // \phi_{10}
    weightResampler->UseReferenceImageOn();
    weightResampler->SetReferenceImage(this->GetFixedImage());

    typedef MultiplyImageFilter<ImageType,ImageType>  ImageMultiplierType;
    typename ImageMultiplierType::Pointer weightMultiplier = ImageMultiplierType::New();
    weightMultiplier->SetInput1(weightResampler->GetOutput()); // W(1) = W_0 \circ \phi_{10}
    weightMultiplier->SetInput2(difference);                   // I(1) - I_1
    weightMultiplier->Update();
      
    difference = weightMultiplier->GetOutput(); // W(1) (I(1) - I_1)
  }

  return 0.5 * vcl_pow( CalculateNorm(difference)/m_Sigma ,2); // 0.5 \sigma^{-2} ||W(1) (I(1) - I_1)||^2
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
GetVelocityEnergy()
{
  return 0.5 * vcl_pow(CalculateNorm(ApplyKernel(m_InverseVelocityKernel,this->m_OutputTransform->GetVelocityField())),2); // 0.5 ||L_V V||^2
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
GetRateEnergy()
{
  if(m_BiasCorrection)
  {
    return 0.5 * vcl_pow( CalculateNorm(ApplyKernel(m_InverseRateKernel,m_Rate))/m_Mu ,2); // 0.5 \mu^{-2} ||L_H \eta||^2
  }
  else
  {
    return 0;
  }
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
double
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
GetEnergy()
{
  if(m_RecalculateEnergy == true)
  {
    m_Energy = GetVelocityEnergy() + GetRateEnergy() + GetImageEnergy(); // E = E_velocity + E_rate + E_image
    m_RecalculateEnergy = false;
  }
  return m_Energy;
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
typename MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::ImagePointer
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
GetBias()   const
{
  return m_Bias.back(); // B(1)
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
InitializeKernels(TimeVaryingImagePointer kernel, TimeVaryingImagePointer inverseKernel, double alpha, double gamma)
{
  /*
  FFTW runs most efficiently when each diminsions size has a small prime factorization.
  This means each diminsion's size has prime factors that are <= 13.
  Therefore we create a kernel satisfying these conditions.
  See "FFT based convolution" by Gaetan Lehmann for more details.
  */
  typename TimeVaryingImageType::IndexType  index = this->m_OutputTransform->GetVelocityField()->GetLargestPossibleRegion().GetIndex();
  typename TimeVaryingImageType::SizeType   size = this->m_OutputTransform->GetVelocityField()->GetLargestPossibleRegion().GetSize();

  // Create list of small prime numbers...
  SizeValueType smallPrimes[] = {2,3,5,7,11,13};

  // For each dimension...
  for(unsigned int i = 0; i < size.GetSizeDimension(); i++)
  {
    // ...while the size[i] doesn't have a small prime factorization...
    while(true)
    {
      // ...test if size[i] has a small prime factarization.
      unsigned int n = size[i];
      for(unsigned int j = 0; j< sizeof(smallPrimes)/sizeof(SizeValueType) && n!=1; j++)
      {
        while(n%smallPrimes[j]==0 && n!=1) { n /= smallPrimes[j];}
      }

      if(n == 1){ break; } // If size[i]'s has a small prime factorization then we're done.
      size[i]++;           // Otherwise increment size[i].
    }
  }

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
    double  A;

    // For every dimension accumulate the sum in A
    unsigned int i;
    for(i = 0, A = gamma; i < ImageDimension; i++)
    {
      A += 2 * alpha * vcl_pow(size[i],2) * ( 1.0-cos(2*vnl_math::pi*k[i]/size[i]) );
    }

    KIt.Set(vcl_pow(A,-2)); // Kernel
    LIt.Set(A);             // "Inverse" kernel
  }
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
Initialize()
{
  // Check parameters
  if(vcl_abs(m_Mu) < 1e-6)    // if mu == 0...
  {
    m_BiasCorrection = false; // ...don't perform bias correction.
  }

  // Initialize velocity, v = 0
  typename FixedImageType::ConstPointer image = this->GetFixedImage();
  typename ImageType::RegionType        imageRegion = image->GetLargestPossibleRegion();
  typename ImageType::IndexType         imageIndex = imageRegion.GetIndex();
  typename ImageType::SizeType          imageSize = imageRegion.GetSize();
  typename ImageType::PointType         imageOrigin = image->GetOrigin();
  typename ImageType::SpacingType       imageSpacing = image->GetSpacing();
  typename ImageType::DirectionType     imageDirection = image->GetDirection();

  typename TimeVaryingFieldType::Pointer velocity = TimeVaryingFieldType::New();

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
    velocityIndex[i] = imageIndex[i];
    velocitySize[i] = imageSize[i];
    velocityOrigin[i] = imageOrigin[i];
    velocitySpacing[i] = imageSpacing[i];
    for(unsigned int j = 0; j < ImageDimension; j++)
    {
      velocityDirection(i,j) = imageDirection(i,j);
    }
  }

  typename TimeVaryingFieldType::RegionType  velocityRegion(velocityIndex,velocitySize);

  velocity->SetRegions(velocityRegion);
  velocity->SetOrigin(velocityOrigin);
  velocity->SetSpacing(velocitySpacing);
  velocity->SetDirection(velocityDirection);
  velocity->Allocate();
  velocity->FillBuffer(NumericTraits<VectorType>::Zero); // v = 0

  // Initialize rate, \eta = 0
  m_Rate->SetRegions(velocityRegion);
  m_Rate->CopyInformation(velocity);
  m_Rate->Allocate();
  m_Rate->FillBuffer(0.0);  // eta = 0;

  // Initialize constants
  m_VoxelVolume = 1;
  for(unsigned int i = 0; i < ImageDimension; i++){ m_VoxelVolume *= imageSpacing[i]; } // \Delta x
  m_NumberOfTimeSteps = velocity->GetLargestPossibleRegion().GetSize()[ImageDimension]; // J
  m_TimeStep = 1.0/(m_NumberOfTimeSteps - 1); // \Delta t

  // Initialize bias, B = 0
  ImagePointer zero = ImageType::New();
  zero->CopyInformation(image);
  zero->SetRegions(imageRegion);
  zero->Allocate();
  zero->FillBuffer(0.0);
  m_Bias = ImageListType(m_NumberOfTimeSteps,zero);

  // Initialize transform
  this->m_OutputTransform->SetVelocityField(velocity);
  this->m_OutputTransform->SetNumberOfIntegrationSteps(m_NumberOfTimeSteps + 2);
  this->m_OutputTransform->SetLowerTimeBound(0.0);
  this->m_OutputTransform->SetUpperTimeBound(1.0);
  this->m_OutputTransform->IntegrateVelocityField();

  // Initialize velocity kernels, K_V, L_V
  InitializeKernels(m_VelocityKernel,m_InverseVelocityKernel,m_RegistrationSmoothness,m_Gamma);

  // Initialize rate kernels, K_H, L_H
  InitializeKernels(m_RateKernel,m_InverseRateKernel,m_BiasSmoothness,m_Gamma);

  m_RecalculateEnergy = true; // v and \eta have been initialized
  m_InitialEnergy = GetEnergy();
  this->InvokeEvent(InitializeEvent());
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
IntegrateBias()
{
  typename TimeVaryingImageType::IndexType  index;
  index.Fill(0);

  typename TimeVaryingImageType::SizeType   size = m_Rate->GetLargestPossibleRegion().GetSize();
  size[ImageDimension] = 0;

  typename TimeVaryingImageType::RegionType region(index,size);

  m_Bias.resize(1);     // Never overwrite first element since we should always have B(0) = 0
  for(unsigned int j = 1; j < m_NumberOfTimeSteps; j++)
  {
    // Calculate B(j) = [\eta(j-1) \Delta t + B(j-1) ] o \phi_{j,j-1}
    index[ImageDimension] = j-1;
    region.SetIndex(index);

    typedef ExtractImageFilter<TimeVaryingImageType,ImageType> ExtractorType;
    typename ExtractorType::Pointer extractor = ExtractorType::New();
    extractor->SetInput(m_Rate);                    // \eta
    extractor->SetExtractionRegion(region);
    extractor->SetDirectionCollapseToIdentity();

    typedef MultiplyImageFilter<ImageType,ImageType> MultiplierType;
    typename MultiplierType::Pointer  multiplier = MultiplierType::New();
    multiplier->SetInput(extractor->GetOutput());   // \eta(j-1)
    multiplier->SetConstant(m_TimeStep);            // \Delta t

    typedef AddImageFilter<ImageType> AdderType;
    typename AdderType::Pointer adder = AdderType::New();
    adder->SetInput1(multiplier->GetOutput());      // \eta(j-1) \Delta t
    adder->SetInput2(m_Bias[j-1]);                  // B(j-1)

    this->m_OutputTransform->SetNumberOfIntegrationSteps(2);
    this->m_OutputTransform->SetLowerTimeBound(j * m_TimeStep);     // t_j
    this->m_OutputTransform->SetUpperTimeBound((j-1) * m_TimeStep); // t_{j-1}
    this->m_OutputTransform->IntegrateVelocityField();

    typedef ResampleImageFilter<ImageType,ImageType,RealType> ResamplerType;
    typename ResamplerType::Pointer  resampler = ResamplerType::New();
    resampler->SetInput(adder->GetOutput());                    // \eta(j-1) \Delta t + B(j-1)
    resampler->SetTransform(this->m_OutputTransform);           // \phi_{j,j-1}
    resampler->UseReferenceImageOn();
    resampler->SetReferenceImage(this->GetFixedImage());
    resampler->Update();

    m_Bias.push_back(resampler->GetOutput());   // B(j) = [\eta(j-1) \Delta t + B(j-1) ] o \phi_{j,j-1}
  }
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
UpdateControls()
{
  typedef JoinSeriesImageFilter<FieldType,TimeVaryingFieldType> FieldJoinerType;
  typename FieldJoinerType::Pointer velocityJoiner = FieldJoinerType::New();

  typedef JoinSeriesImageFilter<ImageType,TimeVaryingImageType> ImageJoinerType;
  typename ImageJoinerType::Pointer rateJoiner = ImageJoinerType::New();

  for(unsigned int j = 0; j < m_NumberOfTimeSteps; j++)
  {
    double t = j * m_TimeStep;
    // Compute forward mapping, \phi_{t0}, by integrating velocity field, v(t).
    if(j == 0)
    {
      this->m_OutputTransform->GetModifiableDisplacementField()->FillBuffer(NumericTraits<VectorType>::Zero);
    }
    else
    {
      this->m_OutputTransform->SetNumberOfIntegrationSteps(j + 2);
      this->m_OutputTransform->SetLowerTimeBound(t);
      this->m_OutputTransform->SetUpperTimeBound(0.0);
      this->m_OutputTransform->IntegrateVelocityField();
    }

    typedef ImageDuplicator<FieldType>  FieldDuplicatorType;
    typename FieldDuplicatorType::Pointer forwardDuplicator = FieldDuplicatorType::New();
    forwardDuplicator->SetInputImage(this->m_OutputTransform->GetDisplacementField());
    forwardDuplicator->Update();

    typedef DisplacementFieldTransform<RealType,ImageDimension> DisplacementFieldTransformType;
    typename DisplacementFieldTransformType::Pointer forwardTransform = DisplacementFieldTransformType::New();
    forwardTransform->SetDisplacementField(forwardDuplicator->GetOutput()); // \phi_{t0}

    // Compute forward image I(t) = I_0 o \phi_{t0}
    typedef ResampleImageFilter<MovingImageType,ImageType,RealType>  MovingResamplerType;
    typename MovingResamplerType::Pointer forwardResampler = MovingResamplerType::New();
    forwardResampler->SetInput(this->GetMovingImage());   // I_0
    forwardResampler->SetTransform(forwardTransform);     // \phi_{t0}
    forwardResampler->UseReferenceImageOn();
    forwardResampler->SetReferenceImage(this->GetFixedImage());

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

    typename FieldDuplicatorType::Pointer backwardDuplicator = FieldDuplicatorType::New();
    backwardDuplicator->SetInputImage(this->m_OutputTransform->GetDisplacementField());
    backwardDuplicator->Update();

    typename DisplacementFieldTransformType::Pointer backwardTransform = DisplacementFieldTransformType::New();
    backwardTransform->SetDisplacementField(backwardDuplicator->GetOutput()); // \phi_{t1}

    // Compute reverse image (I_1 - B(1)) o \phi_{t1}
    typedef SubtractImageFilter<FixedImageType,ImageType,ImageType>  FixedSubtractorType;
    typename FixedSubtractorType::Pointer  fixedSubtractor = FixedSubtractorType::New();
    fixedSubtractor->SetInput1(this->GetFixedImage());  // I_1
    fixedSubtractor->SetInput2(GetBias());              // B(1)

    typedef ResampleImageFilter<ImageType,ImageType,RealType>  FixedResamplerType;
    typename FixedResamplerType::Pointer backwardResampler = FixedResamplerType::New();
    backwardResampler->SetInput(fixedSubtractor->GetOutput());       // I_1 - B(1);
    backwardResampler->SetTransform(backwardTransform);              // \phi_{t1}
    backwardResampler->UseReferenceImageOn();
    backwardResampler->SetReferenceImage(this->GetFixedImage());

    // Compute costate, p(t) = -\sigma^{-2} [ I_0 o \phi_t0 - (I_1 - B(1)) o \phi_t1 ] |D\phi_{t1}|
    typedef SubtractImageFilter<ImageType,ImageType,ImageType>  SubtractorType;
    typename SubtractorType::Pointer subtractor = SubtractorType::New();
    subtractor->SetInput1(forwardResampler->GetOutput());   // I_0 o \phi_t0
    subtractor->SetInput2(backwardResampler->GetOutput());  // (I_1 - B(1)) o \phi_t1

    typedef DisplacementFieldJacobianDeterminantFilter<FieldType,RealType,ImageType>  JacobianDeterminantFilterType;
    typename JacobianDeterminantFilterType::Pointer jacobianDeterminantFilter = JacobianDeterminantFilterType::New();
    jacobianDeterminantFilter->SetInput(backwardDuplicator->GetOutput()); // \phi_{t1}

    typedef MultiplyImageFilter<ImageType,ImageType>  ImageMultiplierType;
    typename ImageMultiplierType::Pointer   multiplier0 = ImageMultiplierType::New();
    multiplier0->SetInput1(subtractor->GetOutput());                // I_0 o \phi_t0 - (I_1 - B(1)) o \phi_t1
    multiplier0->SetInput2(jacobianDeterminantFilter->GetOutput()); // |D\phi_{t1}|

    typename ImageMultiplierType::Pointer   multiplier1 = ImageMultiplierType::New();
    multiplier1->SetInput(multiplier0->GetOutput());  // [ I_0 o \phi_t0 - (I_1 - B(1)) o \phi_t1 ] |D\phi_{t1}|
    multiplier1->SetConstant(-vcl_pow(m_Sigma,-2));   // -\sigma^{-2}
    multiplier1->Update();
    
    typename ImageType::Pointer costate = multiplier1->GetOutput(); // p(t) = -\sigma^{-2} [ I_0 o \phi_t0 - (I_1 - B(1)) o \phi_t1 ] |D\phi_{t1}|
    
    if(this->GetWeightImage().GetPointer() != NULL)
    {
      // Compute p(t) = W(t) * p(t)
      typedef ResampleImageFilter<WeightImageType,ImageType,RealType>  WeightResamplerType;
      typename WeightResamplerType::Pointer weightResampler = WeightResamplerType::New();
      weightResampler->SetInput(this->GetWeightImage());  // W_0
      weightResampler->SetTransform(forwardTransform);    // \phi_{t0}
      weightResampler->UseReferenceImageOn();
      weightResampler->SetReferenceImage(this->GetFixedImage());

      typename ImageMultiplierType::Pointer weightMultiplier = ImageMultiplierType::New();
      weightMultiplier->SetInput1(weightResampler->GetOutput()); // W(t) = W_0 \circ \phi_{t0}
      weightMultiplier->SetInput2(costate);                      // -\sigma^{-2} [ I_0 o \phi_t0 - (I_1 - B(1)) o \phi_t1 ] |D\phi_{t1}|
      weightMultiplier->Update();
      
      costate = weightMultiplier->GetOutput(); // p(t) = -\sigma^{-2} W(t) [ I_0 o \phi_t0 - (I_1 - B(1)) o \phi_t1 ] |D\phi_{t1}|
    }

    // Compute I(t) = I_0 o \phi_t0 + B(t)
    typedef AddImageFilter<ImageType>    AdderType;
    typename AdderType::Pointer movingAdder = AdderType::New();
    movingAdder->SetInput1(forwardResampler->GetOutput()); // I_0 o \phi_t0
    movingAdder->SetInput2(m_Bias[j]);                     // B(t)

    // Compute p(t) \nabla I(t)
    typedef GradientImageFilter<ImageType,RealType,RealType>      GradientFilterType;
    typename GradientFilterType::Pointer gradientFilter = GradientFilterType::New();
    gradientFilter->SetInput(movingAdder->GetOutput());        // I(t) = I_0 o \phi_t0 + B(t)

    typedef VectorCastImageFilter<typename GradientFilterType::OutputImageType, FieldType>  CasterType;
    typename CasterType::Pointer caster = CasterType::New();
    caster->SetInput(gradientFilter->GetOutput());              // \nabla I(t)

    typedef MultiplyImageFilter<ImageType,FieldType,FieldType> FieldMultiplierType;
    typename FieldMultiplierType::Pointer multiplier2 = FieldMultiplierType::New();
    multiplier2->SetInput1(costate);                  // p(t)
    multiplier2->SetInput2(caster->GetOutput());      // \nabla I(t)
    multiplier2->Update();

    rateJoiner->PushBackInput(costate);                       // p(t)
    velocityJoiner->PushBackInput(multiplier2->GetOutput());  // p(t) \nambla I(t)
  } // End time loop
  velocityJoiner->Update();
  rateJoiner->Update();

  // Compute velocity energy gradient, \nabla_V E = v + K_V [p \nambla I]
  TimeVaryingFieldPointer velocityEnergyGradient;
  typedef AddImageFilter<TimeVaryingFieldType> TimeVaryingFieldAdderType;
  typename TimeVaryingFieldAdderType::Pointer adder0 = TimeVaryingFieldAdderType::New();
  adder0->SetInput1(this->m_OutputTransform->GetVelocityField());                 // v
  adder0->SetInput2(ApplyKernel(m_VelocityKernel,velocityJoiner->GetOutput()));   // K_V[p \nambla I]
  adder0->Update();

  velocityEnergyGradient = adder0->GetOutput();                                   // \nabla_V E = v + K_V[p \nambla I]

  // Compute rate energy gradient \nabla_\eta E = \eta - \mu^2 K_H[p]
  TimeVaryingImagePointer rateEnergyGradient;
  typedef MultiplyImageFilter<TimeVaryingImageType,TimeVaryingImageType>  TimeVaryingImageMultiplierType;
  typedef AddImageFilter<TimeVaryingImageType>                            TimeVaryingImageAdderType;
  if(m_BiasCorrection)
  {
    typename TimeVaryingImageMultiplierType::Pointer multiplier1 = TimeVaryingImageMultiplierType::New();
    multiplier1->SetInput(ApplyKernel(m_RateKernel,rateJoiner->GetOutput())); // K_H[p]
    multiplier1->SetConstant(-vcl_pow(m_Mu,2));     // -\mu^2

    typename TimeVaryingImageAdderType::Pointer adder2 = TimeVaryingImageAdderType::New();
    adder2->SetInput1(m_Rate);                     // \eta
    adder2->SetInput2(multiplier1->GetOutput());   // -\mu^2 K_H[p]
    adder2->Update();

    rateEnergyGradient = adder2->GetOutput();      // \nabla_H E = \eta - \mu^2 K_H[p]
  }

  double                  energyOld = GetEnergy();
  TimeVaryingFieldPointer velocityOld = this->m_OutputTransform->GetVelocityField();
  TimeVaryingImagePointer rateOld = m_Rate;

  while(this->GetLearningRate() > m_MinimumLearningRate)
  {
    // Update velocity, v = v - \epsilon \nabla_V E
    typedef MultiplyImageFilter<TimeVaryingFieldType,TimeVaryingImageType>  TimeVaryingFieldMultiplierType;
    typename TimeVaryingFieldMultiplierType::Pointer multiplier = TimeVaryingFieldMultiplierType::New();
    multiplier->SetInput(velocityEnergyGradient);                   // \nabla_V E
    multiplier->SetConstant(-this->GetLearningRate());              // -\epsilon

    typename TimeVaryingFieldAdderType::Pointer adder1 = TimeVaryingFieldAdderType::New();
    adder1->SetInput1(this->m_OutputTransform->GetVelocityField());  // v
    adder1->SetInput2(multiplier->GetOutput());                      // -\epsilon \nabla_V E
    adder1->Update();

    this->m_OutputTransform->SetVelocityField(adder1->GetOutput());  // v = v - \epsilon \nabla_V E

    // Update rate, \eta = \eta - \epsilon \nabla_H E
    if(m_BiasCorrection)
    {
      typename TimeVaryingImageMultiplierType::Pointer multiplier2 = TimeVaryingImageMultiplierType::New();
      multiplier2->SetInput(rateEnergyGradient);            // \nabla_H E
      multiplier2->SetConstant(-this->GetLearningRate());   // -\epsilon

      typename TimeVaryingImageAdderType::Pointer adder3 = TimeVaryingImageAdderType::New();
      adder3->SetInput1(m_Rate);                    // \eta
      adder3->SetInput2(multiplier2->GetOutput());  // -\epsilon \nabla_H E
      adder3->Update();

      m_Rate = adder3->GetOutput(); // \eta = \eta - \epsilon \nabla_H E  */
      IntegrateBias();              // Integrate bias because v and \eta has changed
    }

    m_RecalculateEnergy = true;   // Recalculate energy because v and \eta have changed
    if(GetEnergy() > energyOld)   // If energy increased...
    {
      // ...restore the controls to their previous values and decrease learning rate
      this->SetLearningRate(0.5*this->GetLearningRate());
      this->m_OutputTransform->SetVelocityField(velocityOld);

      if(m_BiasCorrection)
      {
        m_Rate = rateOld;
        IntegrateBias();               // Integrate bias because v and \eta has changed
      }

      m_RecalculateEnergy = true;      // Recalculate energy because v and \eta have changed
    }
    else  // If energy decreased...
    {
      // ...slightly increase learning rate
      this->SetLearningRate(1.05*this->GetLearningRate());
      this->m_IsConverged = false;
      return;
    }

  } // end while

  this->m_IsConverged = true; // Convergence has occured if learning rate is less than minimum learning rate
  return;
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
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

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
GenerateData()
{
  Initialize();
  StartOptimization();

  // Integrate velocity to get final displacement, \phi_10
  this->m_OutputTransform->SetNumberOfIntegrationSteps(m_NumberOfTimeSteps + 2);
  this->m_OutputTransform->SetLowerTimeBound(1);
  this->m_OutputTransform->SetUpperTimeBound(0);
  this->m_OutputTransform->IntegrateVelocityField();

  DecoratedOutputTransformPointer transformDecorator = DecoratedOutputTransformType::New().GetPointer();
  transformDecorator->Set(this->m_OutputTransform);
  this->ProcessObject::SetNthOutput(0, transformDecorator);

  this->InvokeEvent(EndEvent());
}

template<typename TFixedImage, typename TMovingImage, typename TWeightImage>
void
MetamorphosisImageRegistrationMethodv4<TFixedImage, TMovingImage, TWeightImage>::
PrintSelf(std::ostream& os, Indent indent ) const
{
  ProcessObject::PrintSelf(os, indent);
  os<<indent<<"Velocity Smoothness: " <<m_RegistrationSmoothness<<std::endl;
  os<<indent<<"Bias Smoothness: "<<m_BiasSmoothness<<std::endl;
}

} // End namespace itk

#endif
