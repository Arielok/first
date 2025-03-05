from urllib.parse import quote
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os



class ClassBox:
    def __init__(self,mysql_config):
        """
        初始化 ClassBox 类。
        """
        # MySQL 配置
        self.mysql_user = quote(mysql_config['user'])
        self.mysql_password = quote(mysql_config['password'])
        self.mysql_host = quote(mysql_config['host'])
        self.mysql_database = quote(mysql_config['database'])
        self.mysql_port = mysql_config['port']


        # 创建 MySQL 链接字符串、SQLAlchemy 引擎
        self.mysql_connection_string = f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        self.engine = create_engine(self.mysql_connection_string, pool_recycle=3600)

    def _fetch_data(self, query):
            """
            执行 SQL 查询并返回结果
            :param query: SQL 查询语句
            :return: 查询结果的 DataFrame
            """
            df = pd.read_sql(query, self.engine)
            return df

    def get_mysql(self):
            """
            获取毛利数据
            :param start_date: 日期条件查询
            :return: 毛利数据 DataFrame
            """
            query = f"""
                    SELECT date,device_id,ad_id,type,create_time
                    FROM quick_analysis.ods_user_cash_log
                    WHERE date in ('2025-03-01','2025-03-03')  
                           and official_status=10 and brand='huawei' and platform_id=1157 
                    ;
            """
            return self._fetch_data(query)
    def handle_show(self,type_filter):
        df_inner =self.get_mysql()
        df_show = df_inner[df_inner['type']==type_filter]
        df_inner['create_time'] = pd.to_datetime(df_inner['create_time'], unit='ms')
        df_show = df_show.sort_values(by=['date','device_id', 'create_time']) #ascending=False 降序
        df_show['create_time_diff'] = df_show.groupby(['date','device_id'])['create_time'].diff()
        # 将 create_time_diff 除以 1000 并转换为 float 类型保留2位小数
        df_show['create_time_diff'] = df_show['create_time_diff'] / 1000
        df_show['create_time_diff'] = df_show['create_time_diff'].astype(float).round(2)
        df_show = df_show.dropna(subset=['create_time_diff'])
        df_show=df_show[['date','device_id', 'create_time','create_time_diff']]
        return df_show
    def handle_show_click(self):
        df_inner =self.get_mysql()
        df_inner['create_time'] = pd.to_datetime(df_inner['create_time'], unit='ms')
        df_show = df_inner.sort_values(by=['date','device_id','ad_id','create_time'])  # ascending=False 降序
        df_show = df_show.groupby(['date', 'device_id', 'ad_id', 'type']).agg({'create_time': 'max'}).reset_index()
        df_show['create_time_diff'] = df_show.groupby(['date','device_id','ad_id'])['create_time'].diff()
        df_show = df_show.dropna(subset=['create_time_diff'])
        # 将 Timedelta 转换为秒
        df_show['create_time_diff'] = df_show['create_time_diff'].dt.total_seconds()
        df_show = df_show[df_show['create_time_diff'] >= 0]
        df_show=df_show[['date','device_id', 'create_time','create_time_diff']]
        return df_show

    def boxplot(self, df):
        """
        绘制带有特征值标注的箱型图。

        参数:
        df (pd.DataFrame): 输入的 DataFrame，包含 'business_id' 和 'arpu' 列。
        """
        # 按 date 分组
        grouped = df.groupby('date')['create_time_diff']
        # 获取分组标签
        labels = grouped.groups.keys()

        # 遍历每个 date，分别绘制箱型图
        for i, (date, group) in enumerate(grouped):
            # 创建一个单独的箱型图
            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制当前 date 的箱型图
            ax.boxplot(group,showfliers=False)  # showfliers=False
            title = f'[{date}_{define_name}]'
            ax.set_title(title)
            ax.set_xticklabels([str(date)])  # 设置 x 轴标签为 date
            ax.grid(linestyle="--", alpha=0.3)

            # 获取箱型图的特征值
            q1 = np.percentile(group, 25)
            q3 = np.percentile(group, 75)
            iqr = q3 - q1
            median = np.median(group)
            min_value = np.min(group)
            max_value = np.max(group)
            upper_whisker = q3 + 1.5 * iqr
            upper_whisker = round(upper_whisker,2)
            lower_whisker = q1 - 1.5 * iqr

            # 确保须的值不超过数据范围
            upper_whisker = min(upper_whisker, max_value)
            lower_whisker = max(lower_whisker, min_value)

            # 统计异常值
            outliers = group[(group < lower_whisker) | (group > upper_whisker)]
            num_outliers = len(outliers)
            total_num = len(group)
            outlier_ratio = num_outliers / total_num if total_num > 0 else 0
            outlier_ratio = f"{outlier_ratio:.2%}"
            print(f'总数据条数 {total_num}')
            print(f'上须 {upper_whisker}')
            print(f'上四分位(75%):{q3}')
            print(f'中位数:{median}')
            print(f'下四分位(25%):{q1}')
            print(f'下须 {lower_whisker}')

            # 统一标注位置：右下角
            offset_x = 1.2  # x轴的偏移量，向右侧偏移
            offset_y = min_value  # 设置y轴的偏移量，使用min_value作为基准

            # 添加标注（统一展示在右下角）
            # ax.text(offset_x, offset_y + 3.5, f'Max: {max_value:.2f}', fontsize=9, color='orange', ha='left',
            #         va='center')
            # ax.text(offset_x, offset_y + 3, f'Upper Whisker: {upper_whisker:.2f}', fontsize=9, color='purple',
            #         ha='left', va='center')
            # ax.text(offset_x, offset_y + 2.5, f'Q3: {q3:.2f}', fontsize=9, color='blue', ha='left', va='center')
            # ax.text(offset_x, offset_y + 2, f'Median: {median:.2f}', fontsize=9, color='red', ha='left', va='center')
            # ax.text(offset_x, offset_y + 1.5, f'Q1: {q1:.2f}', fontsize=9, color='blue', ha='left', va='center')
            # # ax.text(offset_x, offset_y + 1, f'Min: {min_value:.2f}', fontsize=9, color='orange', ha='left', va='center')
            # ax.text(offset_x, offset_y + 0.5, f'Lower Whisker: {lower_whisker:.2f}', fontsize=9, color='purple',
            #         ha='left', va='center')

            # 如果需要保存每个图像
            # plt.savefig(f"boxplot_business_id_{business_id}.png")
            # plt.savefig(f"D:/data/picture/boxplot_business_id_{business_id}.png")
            # plt.clf()  # 清除当前图像，以便绘制下一个


# 配置
mysql_config = {
    'user': 'quick_analyst',
    'password': 'IGM@2024#quick',
    'host': '101.132.110.125',
    'database': 'quick_analysis',
    'port': 9030
}


# 配置 0展示 1点击
type_filter = 1
define_name = "interval"
# 创建 ClassBox 实例
boxplotter = ClassBox(mysql_config)
df = boxplotter.handle_show(type_filter)
# df = boxplotter.handle_show_click()
print(df.head(10))
# df_1 =boxplotter.boxplot(df)
# plt.show()