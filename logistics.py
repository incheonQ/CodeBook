import pandas 

# 파레토 법칙
# 전체 결과의 80%가 전체 원인의 20%에서 일어나는 현상
def pareto_rules(
        series: pd.Series,
        threshold: float = 0.8,
        )->bool:

    if type(series) != object:
        series = series.apply(lambda x: str(x))
    
    counts = series.value_counts(normalize=True).sort_values(ascending=False)

    cumsum = counts.cumsum()

    index = math.ceil(len(counts) * 0.2)

    value = cumsum.iloc[index-1]

    if value >= threshold:
        return 1
    else:
        return 0


# 소매포화지수 IRS: Index of Retail Saturation
# 소매포화지수 공식: [지역지상의 총가구수 x 가구당 특정 업태에 대한 지출액] / 특정 업태에 대한 총매장 면적
# 값이 클수록 공급보다 수요가 많다는 것을 의미한다.
def irs(
    households: float,
    demand_per_household: float,
    total_area_of_stores: float,
)->float:
    
    # 잠재수요
    potential_demand = households * demand_per_household

    return potential_demand / total_area_of_stores



# 구매력지수 BPI: Buying Power Index
# 구매력지수 공식: (인구비 * 0.2) + (소매 매출액 비 * 0.3) + (유효 구매 소득비 * 0.5)
# 보편적인 가격으로 판매되는 대중 상품의 구매력을 추정하는 경우에는 BPI의 유용성은 높다.
# 하지만, 상품의 성격이 대중 시장을부터 멀수록 보다 많은 차별요소(소득, 계층, 연령, 성별 등)을 가지고 BPI를 수정하여야 한다.
def bpi(
    population_ratio: float,
    retail_sales_ratio: float,
    effective_income_ratio: float,
    weight_population_ratio: float = 0.2,
    weight_retail_sales_ratio: float = 0.3,
    weight_effective_income_ratio: float = 0.5,
)->float:
    return (population_ratio * weight_population_ratio) + (retail_sales_ratio * weight_retail_sales_ratio) + (effective_income_ratio * weight_effective_income_ratio)

    

    
    
