try:
    import numpy as np
    print(f"NumPy版本: {np.__version__}")
    
    import scipy
    print(f"SciPy版本: {scipy.__version__}")
    
    from scipy.spatial import KDTree
    print("成功导入KDTree")
    
    # 创建简单KDTree测试
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    tree = KDTree(points)
    print("成功创建KDTree实例")
    
    import trajdata
    print(f"成功导入trajdata")
    
    from trajdata import UnifiedDataset
    print("成功导入UnifiedDataset")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
