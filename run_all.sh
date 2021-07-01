git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../

pip install -r requirements.txt
chmod +x download.sh
./download.sh

python build_vocab.py   
python resize.py

python train.py

python sample.py --image='png/example.png'