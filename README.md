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
また、誤差逆伝播法の肝である連鎖律を適用することで、出力変数に対する任意の入力変数の微分値が得られる(誤差逆伝播法は連鎖律を適用した二重数の演算の一例であるとも言える)。  
機械学習ライブラリである
[TensorFlow](https://www.tensorflow.org/tutorials/customization/autodiff?hl=ja)
の説明にもある通り、コード中の条件分岐やループ処理などの制御フローも自然に取り扱われる。  
活用例の一つとしては
[微分方程式を深層学習で解く](https://arxiv.org/pdf/1711.10561.pdf)
などがある。
論文ではTensorFlowを利用して時間変数tや位置変数xについての偏微分値を計算し、
[Burgers方程式](https://ja.wikipedia.org/wiki/バーガース方程式)
という衝撃波などを記述する非線形偏微分方程式を満たすようニューラルネットワークに学習させている。    
While studying deep learning, I found the automatic differentiation that is implemented in TensorFlow and dual-number used in that, and made a class to use this.  
I refer to
[Wikipedia](https://en.wikipedia.org/wiki/Dual_number)
and so on to learn dual-number.  
The following is a memorandum...  
The dual-number is a concept similar to the complex-number, which introduces a new element that meets zero when squared.
The most merit of using this into machine learning is that the differential value appears in the part of the imaginary dual part, so that it can be obtained without paying special attention to the differential value of each operation.  
Also, by applying the chain rule, which is the core of the error backpropagation, the differential value of an arbitrary input variable with respect to the output variable can be obtained (It can be said that the error backpropagation is an example of a dual-number operation to which the chain rule is applied).  
As described in 
[TensorFlow](https://www.tensorflow.org/tutorials/customization/autodiff)
, a machine learning library, control flows such as conditional branching and loop processing in the code are also handled naturally.  
One of application example is
[Solving differential equations by deep learning](https://arxiv.org/pdf/1711.10561.pdf)
.
In this paper, TensorFlow is used to calculate partial differential values for time variable 't' and position variable 'x', and neural networks are trained to satisfy
[Burgers equation](https://en.wikipedia.org/wiki/Burgers%27_equation)
, which is a nonlinear partial differential equation that describes shock waves and so on.

## VS
[tmurakami1234さんのモジュール](https://github.com/tmurakami1234/my_python_module/tree/master/dual)
を出発点の参考とさせていただき、
[Pythonの特殊メソッド](https://docs.python.org/ja/3/reference/datamodel.html)
をNumpyに対応させました。    

I used
[Mr. tmurakami1234's module]()
as a starting point and made
[Python special method](https://docs.python.org/3/reference/datamodel.html)
compatible with Numpy.

## Demo
    import dual as d
    x = Dual(1, 2)
    y = Dual([1, 2], [3, 4])
    z = Dual(2, 1)
    print(x * y)              # Dual([1, 2], [5, 8])
    print(x / y)              # Dual([1, 0.5], [-1, 0])
    print(z ** 3 - z)         # Dual(6, 11)
    # 微分3z^2-1にz=2を代入し1を掛けた値11が虚二重数部に現れている。
    # The value 11 obtained by substituting z=2 into the derivative 3z^2-1 and multiplying by 1 appears in the imaginary dual part.
などなど。
詳しくは
[テストコード](https://github.com/YumaShimomoto/dual/blob/master/dual/dual_test.py)
を参考にしてください。    

""" code lines """
and so on.
For details, refer to the
[test code](https://github.com/YumaShimomoto/dual/blob/master/dual/dual_test.py)
.

## Feature
* Pythonで書かれた二重数のコード
* Numpyに対応(のつもり)
  
* Dual-number code written in Python
* Support Numpy (I think I did...)

## Requirement
* Python3.5 ~
* Numpy 1.10 ~

## TODO
* Numpyに実装されている数学関数やユーティリティ関数を実装
* READMEの拡充
  
* Implements mathematical and utility functions implemented in Numpy
* Expansion of README

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
