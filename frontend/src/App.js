import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid
} from "recharts";

function App() {
  const [stocks, setStocks] = useState({});
  const [selected, setSelected] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/stocks")
      .then(res => setStocks(res.data));
  }, []);

  const handlePredict = async () => {
    const response = await axios.get(
      `http://127.0.0.1:8000/predict/${selected}`
    );
    setResult(response.data);
  };

  return (
    <div style={{ padding: "40px" }}>
      <h1>AI Stock Prediction Dashboard</h1>

      <select onChange={(e) => setSelected(e.target.value)}>
        <option>Select Company</option>
        {Object.entries(stocks).map(([name, symbol]) => (
          <option key={symbol} value={symbol}>
            {name}
          </option>
        ))}
      </select>

      <button onClick={handlePredict}>Predict</button>

      {result && (
        <div>
          <h3>Current Price: ₹{result.current_price}</h3>
          <h3>Predicted Price: ₹{result.predicted_price}</h3>

          <LineChart width={600} height={300}
            data={result.history.map((price, index) => ({
              day: index,
              price: price
            }))}>
            <CartesianGrid stroke="#ccc" />
            <XAxis dataKey="day" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="price" />
          </LineChart>
        </div>
      )}
    </div>
  );
}

export default App;