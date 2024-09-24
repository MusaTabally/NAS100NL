#include <iostream>
#include <vector>

// A simple backtesting framework for NASDAQ futures trading
class Backtest {
public:
  // Input: historical prices, model predictions (buy/sell signals)
  Backtest(const std::vector<float>& prices, const std::vector<int>& signals)
      : prices(prices), signals(signals), initial_capital(100000), capital(100000), position(0) {}

  // Run the backtest simulation
  void run() {
    for (size_t i = 1; i < prices.size(); ++i) {
      if (signals[i] == 1) {  // Buy signal
        buy(i);
      } else if (signals[i] == -1 && position > 0) {  // Sell signal
        sell(i);
      }
      update_capital(i);
    }
    report();
  }

private:
  std::vector<float> prices;
  std::vector<int> signals;  // 1 = Buy, -1 = Sell, 0 = Hold
  float initial_capital;
  float capital;
  float position;

  void buy(size_t index) {
    // Buy all available capital worth of NASDAQ futures
    position = capital / prices[index];
    capital = 0;
  }

  void sell(size_t index) {
    // Sell the current position
    capital = position * prices[index];
    position = 0;
  }

  void update_capital(size_t index) {
    // If holding position, update unrealized P&L
    if (position > 0) {
      capital = position * prices[index];
    }
  }

  void report() {
    float profit = capital - initial_capital;
    std::cout << "Initial Capital: " << initial_capital << std::endl;
    std::cout << "Final Capital: " << capital << std::endl;
    std::cout << "Total Profit: " << profit << std::endl;
    std::cout << "Return on Investment: " << (profit / initial_capital) * 100 << "%" << std::endl;
  }
};

int main() {
  // Historical prices for NASDAQ-100 futures (dummy data)
  std::vector<float> prices = {12000, 12100, 12200, 12300, 12250, 12150, 12350, 12400, 12300, 12450};
  
  // Example buy/sell signals (1 = Buy, -1 = Sell, 0 = Hold)
  std::vector<int> signals = {0, 1, 0, 0, -1, 0, 1, 0, 0, -1};

  // Create backtest instance and run
  Backtest backtest(prices, signals);
  backtest.run();

  return 0;
}
