#pragma once

#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <algorithm>
class BMP
{
public:
	class RGB
	{
	public:
		unsigned char red;
		unsigned char green;
		unsigned char blue;
		unsigned char reserved = 0;
		bool operator ==(const RGB r) const noexcept
		{
			return this->red == r.red && this->green == r.green && this->blue == r.blue;
		}
		bool operator !=(const RGB r) const noexcept
		{
			return this->red != r.red && this->green != r.green && this->blue != r.blue;
		}
		bool operator <(const RGB r) const noexcept
		{
			return (this->red < r.red)
				|| (this->red == r.red && this->green < r.green)
				|| (this->red == r.red && this->green == r.green && this->blue < r.blue);

		}
		void operator =(const RGB r)
		{
			this->red = r.red;
			this->green = r.green;
			this->blue = r.blue;
		}
		int convertRGBtoINT()
		{
			int color = 0;
			color |= this->red | this->green << 8 | this->blue << 16;
			return color;
		}
		static RGB convertINTtoRGB(int x)
		{
			RGB color;
			int mask = 255;
			color.red = x & mask;
			color.green = (x & (mask << 8))>>8;
			color.blue = (x & (mask << 16))>>16;
			return color;
		}
	};
	BITMAPFILEHEADER image_file;
	BITMAPINFOHEADER image_info;
	std::vector<RGB> h_all_colors;
	std::vector<RGB> h_color_palette;
	std::vector<UINT8> h_all_colors_resize;
	std::string filename;
	std::chrono::milliseconds elapsed_palette;
	std::chrono::milliseconds elapsed_applying;
bool checkTable(RGB color, const std::vector<RGB>& table, size_t& index)
	{
		for (int i = 0; i < index; i++)
		{
			if (table[i] == color)
				return false;
		}
		return true;
	}
bool collectAllColors()
{
	FILE* bmpFile = fopen(filename.data(), "rb");
	if (bmpFile)
	{
		//READ IMAGE HEADERS
		fread(&image_file, sizeof(image_file), 1, bmpFile);
		fread(&image_info, sizeof(image_info), 1, bmpFile);

		h_all_colors.resize(image_info.biHeight * image_info.biWidth);
		const int padding = (4 - (image_info.biWidth * 3) % 4) % 4;//calculate padding for align to 4 bytes
		for (int i = 0; i < image_info.biHeight; ++i)//read colors 
		{
			for (int j = 0; j < image_info.biWidth; ++j)
			{
				fread(&h_all_colors[i * image_info.biWidth + j].blue, sizeof(h_all_colors[i * image_info.biWidth + j].blue), 1, bmpFile);
				fread(&h_all_colors[i * image_info.biWidth + j].green, sizeof(h_all_colors[i * image_info.biWidth + j].green), 1, bmpFile);
				fread(&h_all_colors[i * image_info.biWidth + j].red, sizeof(h_all_colors[i * image_info.biWidth + j].red), 1, bmpFile);
			}
			fseek(bmpFile, padding, SEEK_CUR);//padding
		}
		fclose(bmpFile);
		return true;
	}
	return false;
}
void WriteBMP8(std::string name)
{
	std::ofstream file(name,std::ofstream::binary);
	
	image_info.biClrUsed = h_color_palette.size();
	image_info.biBitCount = 8;
	image_info.biSizeImage = h_all_colors_resize.size()*sizeof(UINT8);
	image_file.bfSize = sizeof(BITMAPFILEHEADER) + image_info.biSize + h_color_palette.size()*sizeof(RGBQUAD)+ image_info.biSizeImage;
	image_file.bfOffBits =  sizeof(BITMAPFILEHEADER) +	image_info.biSize + image_info.biClrUsed * sizeof(RGBQUAD);
	unsigned char reserved = 0;
	const int padding = (4 - ((4 - (image_info.biWidth * 3) % 4) % 4))&3;
		file.write(reinterpret_cast<char*>(&image_file), sizeof(image_file));
		file.write(reinterpret_cast<char*>(&image_info), sizeof(image_info));
		for (auto& v : h_color_palette)
		{
			file.write(reinterpret_cast<char*>(&v.blue), sizeof(UINT8));
			file.write(reinterpret_cast<char*>(&v.green), sizeof(UINT8));
			file.write(reinterpret_cast<char*>(&v.red), sizeof(UINT8));
			file.write(reinterpret_cast<char*>(&v.reserved), sizeof(UINT8));
		}
		for (int i = 0; i < image_info.biHeight; i++)
		{
			for (int j = 0; j < image_info.biWidth; j++)
			{
				file.write(reinterpret_cast<char*>(&h_all_colors_resize[i * image_info.biWidth + j]), sizeof(UINT8));
			}
			for(int k=0;k<padding;k++)
			file.write(reinterpret_cast<char*>(&reserved), sizeof(UINT8));
		}
		
	file.close();
}
void returnColors(std::vector<RGB> &result)
{
	result.resize(h_all_colors_resize.size());
	int i = 0;
	for (auto v : h_all_colors_resize)
	{
		result[i] = h_color_palette[v];
		i++;
	}
}
void h_createColorPallete()
{
	size_t index = 0;
	auto begin = std::chrono::steady_clock::now();
	for (auto v : h_all_colors)
	{
		if (checkTable(v, h_color_palette, index))
		{
			h_color_palette.push_back(v);
			index++;
		}
	}
	auto end = std::chrono::steady_clock::now();
	elapsed_palette = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
	
}
int findColor(RGB color)
{
	int i = 0;
	for (auto v : h_color_palette)
	{
		if (v == color)
			return i;
		i++;
	}
	return i;
}
void h_applyPalette()
{
	h_all_colors_resize.resize(h_all_colors.size());
	int i = 0;
	auto begin = std::chrono::steady_clock::now();
	for (auto v : h_all_colors)
	{
		h_all_colors_resize[i] = findColor(v);
		i++;
	}
	auto end = std::chrono::steady_clock::now();
	elapsed_applying = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
}
void writeHTML()
{
	char h = '"';
	std::ofstream to;
	to.open("000.html", std::ios_base::out);
	to << u8"<html><head><meta charset='utf-8'></head><body>";
	int i = 0;
	for (auto v : h_all_colors)
	{
		to << u8"<a style=" << h << "background-color:" << "rgb(" << (int)v.red << "," << (int)v.green << "," << (int)v.blue << ");" << h << ">" << "a" << u8"</a>";
		i++;
		if (i % image_info.biWidth == 0)
			to << "<br>";
	}
	to<< u8"</body></html>";
	to.close();
}
void printAllColors()
{
	for (auto v : h_all_colors)
		std::cout << (int)v.red << " " << (int)v.green << " " << (int)v.blue << "\n";
}
void printAllColorsResize()
{
	for (auto v : h_all_colors_resize)
		std::cout << v << " ";
}
void printColorPallete()
{
	size_t i = 0;
	for (auto v : h_color_palette)
	{
		std::cout << i <<". " << (int)v.red << " " << (int)v.green << " " << (int)v.blue << "\n";
		i++;
	}
}
bool rfile(std::string file)
{
	filename = file;
	return collectAllColors();
}
BMP(std::string file)
{

	filename = file;
}
BMP()
{
}
};

