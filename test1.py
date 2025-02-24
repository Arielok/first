print("hello hub")

# ------------------------------------【盒须图初版】--------------------------------
from urllib.parse import quote
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ClassBox:
    def __init__(self,mysql_config,mysql_config2):
        """
        初始化 ClassBox 类。
        """
        # MySQL 配置
        self.mysql_user = quote(mysql_config['user'])
        self.mysql_password = quote(mysql_config['password'])
        self.mysql_host = quote(mysql_config['host'])
        self.mysql_database = quote(mysql_config['database'])
        self.mysql_port = mysql_config['port']

        # doris 配置
        self.mysql_user2 = quote(mysql_config2['user'])
        self.mysql_password2 = quote(mysql_config2['password'])
        self.mysql_host2 = quote(mysql_config2['host'])
        self.mysql_database2 = quote(mysql_config2['database'])
        self.mysql_port2 = mysql_config2['port']

        # 创建 MySQL 链接字符串、SQLAlchemy 引擎
        self.mysql_connection_string = f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        self.engine = create_engine(self.mysql_connection_string, pool_recycle=3600)
        # 创建 doris 链接字符串、SQLAlchemy 引擎
        self.mysql_connection_string2 = f"mysql+pymysql://{self.mysql_user2}:{self.mysql_password2}@{self.mysql_host2}:{self.mysql_port2}/{self.mysql_database2}"
        self.engine_doris = create_engine(self.mysql_connection_string2, pool_recycle=3600)

    def _fetch_data(self, query):
            """
            执行 SQL 查询并返回结果
            :param query: SQL 查询语句
            :return: 查询结果的 DataFrame
            """
            df = pd.read_sql(query, self.engine)
            return df

    def get_mysql(self, start_date,end_date):
            """
            获取毛利数据
            :param start_date: 日期条件查询
            :return: 毛利数据 DataFrame
            """
            query = f"""
                    SELECT businessId as business_id, nickname ,sum(parentProfit) as total_parentProfit
                    FROM dws_businessStat_di
                    WHERE date between "{start_date}" and "{end_date}" 
                    group by businessId, nickname
                    having total_parentProfit > 100000;
            """
            return self._fetch_data(query)

    def _fetch_data_doris(self, query):
            """
            执行 SQL 查询并返回结果
            :param query: SQL 查询语句
            :return: 查询结果的 DataFrame
            """
            df = pd.read_sql(query, self.engine_doris)
            return df

    def get_doris(self, start_date,end_date,business_targe_list):
            """
            获取毛利数据
            :param start_date: 日期条件查询
            :return: 毛利数据 DataFrame
            """
            query = f"""
                        select type,date,business_id,device_id,sum(use_ecpm)/100000 as device_arpu
                        from ods_user_cash_log
                        where type =0 and brand in('honor', 'xiaomi', 'oppo', 'vivo', 'huawei')
                              and date between "{start_date}" and "{end_date}"  and business_id in {business_targe_list}
                        group by type,date,business_id,device_id;
            """
            return self._fetch_data_doris(query)

    def get_data_merge(self):
        business_targe = self.get_mysql(start_date, end_date)
        business_targe_list = tuple(business_targe['business_id'].tolist())
        user_cash_data = self.get_doris(start_date,end_date,business_targe_list)
        final_data = user_cash_data.merge(
            business_targe,
            on=['business_id'], how='left')
        return final_data

    def boxplot(self, df):
        """
        绘制带有特征值标注的箱型图。

        参数:
        df (pd.DataFrame): 输入的 DataFrame，包含 'business_id' 和 'arpu' 列。
        """
        # 按 business_id 分组
        grouped = df.groupby('business_id')['arpu']
        # 获取分组标签
        labels = grouped.groups.keys()

        # 遍历每个 business_id，分别绘制箱型图
        for i, (business_id, group) in enumerate(grouped):
            # 创建一个单独的箱型图
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制当前 business_id 的箱型图
            ax.boxplot(group)
            ax.set_title(f'Boxplot for Business ID {business_id}')
            ax.set_xticklabels([str(business_id)])  # 设置 x 轴标签为 business_id
            ax.grid(linestyle="--", alpha=0.3)

            # 获取箱型图的特征值
            q1 = np.percentile(group, 25)
            q3 = np.percentile(group, 75)
            iqr = q3 - q1
            median = np.median(group)
            min_value = np.min(group)
            max_value = np.max(group)
            upper_whisker = q3 + 1.5 * iqr
            lower_whisker = q1 - 1.5 * iqr

            # 确保须的值不超过数据范围
            upper_whisker = min(upper_whisker, max_value)
            lower_whisker = max(lower_whisker, min_value)

            # 统一标注位置：右下角
            offset_x = 1.2  # x轴的偏移量，向右侧偏移
            offset_y = min_value  # 设置y轴的偏移量，使用min_value作为基准

            # 添加标注（统一展示在右下角）
            ax.text(offset_x, offset_y + 1.7, f'Max: {max_value:.2f}', fontsize=9, color='orange', ha='left', va='center')
            ax.text(offset_x, offset_y + 1.5, f'Upper Whisker: {upper_whisker:.2f}', fontsize=9, color='purple',ha='left', va='center')
            ax.text(offset_x, offset_y + 1.3, f'Q3: {q3:.2f}', fontsize=9, color='blue', ha='left', va='center')
            ax.text(offset_x, offset_y + 1.1, f'Median: {median:.2f}', fontsize=9, color='red', ha='left', va='center')
            ax.text(offset_x, offset_y + 0.9, f'Q1: {q1:.2f}', fontsize=9, color='blue', ha='left', va='center')
            ax.text(offset_x, offset_y + 0.7, f'Min: {min_value:.2f}', fontsize=9, color='orange', ha='left', va='center')
            ax.text(offset_x, offset_y + 0.5, f'Lower Whisker: {lower_whisker:.2f}', fontsize=9, color='purple', ha='left', va='center')

            # 如果需要保存每个图像
            # plt.savefig(f"boxplot_business_id_{business_id}.png")
            plt.savefig(f"D:/data/picture/boxplot_business_id_{business_id}.png")
            # plt.clf()  # 清除当前图像，以便绘制下一个

# 配置
mysql_config = {
    'user': 'analyuser',
    'password': 'IGM@2025@analysql!123',
    'host': '139.196.214.14',
    'database': 'analy',
    'port': 3306
}
mysql_config2 = {
    'user': 'quick_analyst',
    'password': 'IGM@2024#quick',
    'host': '101.132.110.125',
    'database': 'quick_analysis',
    'port': 9030
}

start_date = '2025-02-18'
end_date = '2025-02-18'

# 创建 ClassBox 实例
boxplotter = ClassBox(mysql_config,mysql_config2)
df = boxplotter.get_data_merge()
df = df.filter(['business_id','device_arpu']).rename(columns={'device_arpu': 'arpu'})

# 指定要保留的 business_id 值
selected_business_ids = [57450, 41131]
# 使用布尔索引过滤 DataFrame
df = df[df['business_id'].isin(selected_business_ids)]

print(df.head(3))
result = boxplotter.boxplot(df)
plt.show()


