from pathlib import Path
import kagglehub

def check_datasset():
      path = kagglehub.dataset_download("alexattia/the-simpsons-characters-dataset")
      data_dir = Path(path) /"simpsons_dataset"
      print(f"📂 Путь к данным: {data_dir}")
    #   for item in data_dir.iterdir():
    #     print(f"   {item.name} {'(папка)' if item.is_dir() else '(файл)'}")
      if not data_dir.exists():
        print("❌ Ошибка: Папка не найдена!")
        return
      exclude_names = {'simpsons_dataset', 'test', 'temp'}
      classes = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name not in exclude_names]
      print(f"Найдено классов {(classes)}")
      
      total_images = 0 
      for clss in classes:
          images = len(list((data_dir/clss).glob("*.jpg")))
          total_images+=images
      print(f"Всего изображений{total_images}")
if __name__ == "__main__":
     check_datasset()