from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCharts import QChart, QChartView, QLineSeries
from PyQt6.QtGui import QPainter

class TestChart(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test PyQt6-Charts")
        self.setGeometry(100, 100, 800, 600)

        chart = QChart()
        chart.setTitle("Simple Line Chart")

        series = QLineSeries()
        series.append(0, 0)
        series.append(1, 1)
        series.append(2, 4)
        series.append(3, 9)
        series.append(4, 16)

        chart.addSeries(series)
        chart.createDefaultAxes()

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        self.setCentralWidget(chart_view)

if __name__ == "__main__":
    app = QApplication([])
    window = TestChart()
    window.show()
    app.exec()