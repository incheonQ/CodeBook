from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Regressor:
    
    def __init__(self):
        self.model = None
        self.y_pred = None
        self.best_params = None

    # 준비된 데이터 입력
    def data(self, X, y):
        self.X = X
        self.y = y
        
    # 데이터셋 분리
    def separate(self, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
    
    # 단순 선형 회귀
    def simple_linear(self):
        print('[Simple Linear Regression]')
        self.model = LinearRegression()
        self._fit_and_predict()
    
    # 의사 결정 트리
    def decision_tree(self):
        print('[Decision Tree]')
        self.model = DecisionTreeRegressor()
        self._fit_and_predict()
    
    # 랜덤 포레스트
    def random_forest(self):
        print('[Random Forest]')
        self.model = RandomForestRegressor()
        self._fit_and_predict()
        
    # 그래디언트 부스팅
    def gradient_boosting(self):
        print('[Gradient Boosting]')
        self.model = GradientBoostingRegressor()
        self._fit_and_predict()
    
    # 모델 학습 및 예측
    def _fit_and_predict(self):
        if self.best_params:
            self.model.set_params(**self.best_params)
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
    
    # 하이퍼파라미터 튜닝 (Grid Search)
    def hyperparameter_tuning(self, param_grid, cv=5):
        print(f'[Hyperparameter Tuning with {cv}-fold Cross-Validation]')

        grid_search = GridSearchCV(
            estimator=self.model, 
            param_grid=param_grid, 
            cv=cv,
            scoring='neg_mean_squared_error', 
            verbose=0,  
            n_jobs=-1
        )

        print("Starting Grid Search...")
        grid_search.fit(self.X_train, self.y_train)

        self.best_params = grid_search.best_params_
        print(f"\nBest Parameters: {self.best_params}")
        print(f"Best Score: {-grid_search.best_score_:.4f} MSE")

        self.model = grid_search.best_estimator_

        # 최적의 모델로 예측 및 R2 계산
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"\nTest Set Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2 Score: {r2:.4f}")

        print("\nTop 5 Parameter Combinations:")
        results = grid_search.cv_results_
        for i in range(min(5, len(results['params']))):
            print(f"Rank {i+1}:")
            print(f"  Parameters: {results['params'][i]}")
            print(f"  Mean Score: {-results['mean_test_score'][i]:.4f} MSE")
            print(f"  Std Score: {results['std_test_score'][i]:.4f}")
            print()

        print("Hyperparameter tuning completed.")

    # 교차 검증
    def cross_validate(self, cv=5):
        print(f'[Cross-Validation with {cv}-fold]')
        scores = cross_validate(self.model, self.X, self.y, cv=cv,
                                scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
        mse_scores = -scores['test_neg_mean_squared_error']
        mae_scores = -scores['test_neg_mean_absolute_error']
        r2_scores = scores['test_r2']
        
        print(f"Cross-Validated Mean Squared Error (MSE): {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")
        print(f"Cross-Validated Mean Absolute Error (MAE): {mae_scores.mean():.2f} ± {mae_scores.std():.2f}")
        print(f"Cross-Validated R^2 Score: {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")
    
    # 회귀 모델 성능 평가
    def evaluation(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)

        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"R^2 Score: {r2:.2f}")

        # 잔차 분석
        residuals = self.y_test - self.y_pred

        # 잔차 플롯
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(self.y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')

        plt.subplot(1, 2, 2)
        sns.histplot(residuals, kde=True, color='blue')
        plt.xlabel('Residuals')
        plt.title('Distribution of Residuals')

        plt.tight_layout()
        plt.show()
