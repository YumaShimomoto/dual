# 二重数 Dual number
二重数を扱うためのクラス。  
To use dual-number.

## Disctiption
深層学習について勉強している際、TensorFlowに実装されているという自動微分およびその実装に使われる二重数について知り、これを扱うためのクラスを作成してみました。  
二重数については
[Wikipedia](https://ja.wikipedia.org/wiki/二重数)
などを参考にしています。  
以下、覚書。  
二重数とは二乗するとゼロとなる新たな元を導入した、複素数と似た概念である。
これを機械学習に導入する最大のメリットは、微分値が虚二重数の部分に現れるために各演算の微分値を特に意識することなく得られることである。  
また、誤差逆伝播法の肝である連鎖律を適用することで、出力変数に対する任意の入力変数の微分値が得られる(誤差逆伝播法は連鎖律を適用した限定的な二重数の演算であるとも言える)。  
機械学習ライブラリである
[TensorFlow](https://www.tensorflow.org/tutorials/customization/autodiff?hl=ja)
の説明にもある通り、コード中の条件分岐やループ処理などの制御フローも自然に取り扱われる。  
活用例の一つとしては
[微分方程式を深層学習で解く](https://arxiv.org/pdf/1711.10561.pdf)
などがある。
論文ではTensorFlowを利用して時間変数tや位置変数xについての偏微分値を計算し、
[Burgers方程式](https://ja.wikipedia.org/wiki/バーガース方程式)という衝撃波などを記述する非線形偏微分方程式を満たすようニューラルネットワークに学習させている。

## VS
[tmurakami1234さんのモジュール](https://github.com/tmurakami1234/my_python_module/tree/master/dual)
を出発点の参考とさせていただき、
[Pythonの特殊メソッド](https://docs.python.org/ja/3/reference/datamodel.html)
をNumpyに対応させました。

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
