#ifndef __itkDebug_h
#define __itkDebug_h
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkTimeProbe.h"

template<typename TImage>
void printImage(typename TImage::Pointer image)
{
  std::cout<<"Origin = "<<image->GetOrigin()<<std::endl;
  std::cout<<"Spacing = "<<image->GetSpacing()<<std::endl;
  std::cout<<"Direction = "<<std::endl<<image->GetDirection();
  std::cout<<"Size = "<< image->GetLargestPossibleRegion().GetSize()<<std::endl;
  std::cout<<"Start = "<<image->GetLargestPossibleRegion().GetIndex()<<std::endl;
  std::cout<<"Data = "<<std::endl;

  typedef typename TImage::PixelType PixelType;
  typename TImage::SizeType size = image->GetLargestPossibleRegion().GetSize();
  for(unsigned int j = 0; j < size[1]; j++)
  {
    for(unsigned int i = 0; i < size[0]; i++)
    {
      typename TImage::IndexType index = {{i, j}};
      PixelType pixel = image->GetPixel(index);
      
      /*
      if(typeid(PixelType).name() == typeid(unsigned char).name())
      {
        std::cout<<float(pixel)<<"\t";        
      }
      else
      */
      {
        std::cout<<pixel<<"\t";        
      }

    }
    std::cout<<std::endl;
  }



}


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

template<typename TVar>
std::string toString(TVar var)
{
  std::stringstream ss;
  ss << var;
  return ss.str();
}

using namespace std;
#endif
