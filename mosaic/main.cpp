#include <iostream>
#include <iomanip>
#include <limits>

#include "src/mosaic/mosaic.h"

int main() {
  std::string tile_path, image_path;
  int color_depth{};
  std::vector<double> scales{};
  cv::Size shift{};
  int choice;

  std::cout << std::setfill('-') << std::setw(80) << "" << std::endl;
  std::cout << std::setfill(' ') << std::setw(40) << "Mosaic++" << std::setw(40) << std::endl;
  std::cout << std::setfill('-') << std::setw(80) << "" << std::endl;
  std::cout << "Если вы хотите использовать стандартные гиперпараметры, введите 0\n"
               "Если вы хотите использовать свои параметры, введите любое другое число: ";
  std::cin >> choice;
  if (choice == 0) {
    color_depth = 16;
    scales = {0.25, 0.2, 0.15};
    shift = {};
    tile_path = "../res/heart.png";
    image_path = "../res/kitten_2.png";
  } else {
    std::cout << "Введите глубину цвета плитки: ";
    std::cin >> color_depth;
    std::cout << "Введите коэффициенты масштабирования (0 чтобы закончить ввод):" << std::endl;
    double buffer;
    while ((cin >> buffer) && (std::fabs(buffer - 0) <= std::numeric_limits<double>::min()))
      scales.push_back(buffer);

    if (scales.empty())
      scales.push_back(1);

    int buffer_1, buffer_2;
    std::cout << "Введите отступ (два целых числа через пробел, или два нуля для полного замощения): ";
    std::cin >> buffer_1 >> buffer_2;
    if ((buffer_1 == 0) && (buffer_2 == 0))
      shift = {buffer_1, buffer_2};

    std::cout << "Введите путь к изображению-шаблону: ";
    std::cin >> tile_path;
    std::cout << "Введите путь к изображению, которое хотите обработать: ";
    std::cin >> image_path;
  }

  std::cout << "Данные получены, ожидайте..." << std::endl;
  MosaicGenerator mosaic_generator = MosaicGenerator(color_depth, scales, shift);
  cv::Mat result = mosaic_generator.GenerateMosaic(tile_path, image_path);
  std::cout << "Изображение готово!" << std::endl;

  imshow("result", result);
  waitKey();

  std::cout << "Сохраняем изображение, ожидайте..." << std::endl;
  imwrite("result.png", result);
  std::cout << "Изображение сохранено, файл result.png" << std::endl;

  return 0;
}
