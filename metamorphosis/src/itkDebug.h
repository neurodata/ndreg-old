#ifndef __itkDebug_h
#define __itkDebug_h
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"

template<typename TImage>
void writeImage(typename TImage::Pointer image, std::string path)
{
  typedef itk::ImageFileWriter<TImage> WriterType;
  typename WriterType::Pointer writer = WriterType::New();
  writer->SetInput(image);
  writer->SetFileName(path);
  try
  {
    writer->Update();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    std::cerr<<"Could not write image: "<<path<<std::endl;
    std::cerr<<exceptionObject<<std::endl;
  }
}

template<typename TImage>
typename TImage::Pointer readImage(std::string path)
{
  typedef itk::ImageFileReader<TImage> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(path);
  try
  {
    reader->Update();
    return reader->GetOutput();
  }
  catch(itk::ExceptionObject& exceptionObject)
  {
    std::cerr<<"Could not read image: "<<path<<std::endl;
    std::cerr<<exceptionObject<<std::endl;
    return NULL;
  }
}

template<typename TInput>
void print(TInput s)
{
 std::cout<<s<<std::endl;
}

#endif
