import { useState } from 'react'
import axios from 'axios'

function App() {
  const [formData, setFormData] = useState({
    Type: 'L',
    Air_temperature_K: 300.0,
    Process_temperature_K: 310.0,
    Rotational_speed_rpm: 1500,
    Torque_Nm: 40.0,
    Tool_wear_min: 0
  })

  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData({
      ...formData,
      [name]: name === 'Type' ? value : parseFloat(value)
    })
  }

  const fillRandom = () => {
    const types = ['L', 'M', 'H']
    setFormData({
      Type: types[Math.floor(Math.random() * types.length)],
      Air_temperature_K: parseFloat((Math.random() * (305 - 295) + 295).toFixed(1)),
      Process_temperature_K: parseFloat((Math.random() * (315 - 305) + 305).toFixed(1)),
      Rotational_speed_rpm: Math.floor(Math.random() * (2800 - 1200) + 1200),
      Torque_Nm: parseFloat((Math.random() * (75 - 10) + 10).toFixed(1)),
      Tool_wear_min: Math.floor(Math.random() * 250)
    })
    setPrediction(null)
    setError(null)
  }

  const predict = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post('http://localhost:8000/predict', formData)
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to connect to API. Is the server running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <div className="card">
        <h1>Predictive Maintenance</h1>
        <p className="subtitle">Random Forest Model Testing Interface</p>

        <div className="form-grid">
          <div className="form-group">
            <label>Product Type</label>
            <select name="Type" value={formData.Type} onChange={handleInputChange}>
              <option value="L">L (Low)</option>
              <option value="M">M (Medium)</option>
              <option value="H">H (High)</option>
            </select>
          </div>

          <div className="form-group">
            <label>Air Temp [K]</label>
            <input
              type="number"
              name="Air_temperature_K"
              value={formData.Air_temperature_K}
              onChange={handleInputChange}
              step="0.1"
            />
          </div>

          <div className="form-group">
            <label>Process Temp [K]</label>
            <input
              type="number"
              name="Process_temperature_K"
              value={formData.Process_temperature_K}
              onChange={handleInputChange}
              step="0.1"
            />
          </div>

          <div className="form-group">
            <label>Rotational Speed [rpm]</label>
            <input
              type="number"
              name="Rotational_speed_rpm"
              value={formData.Rotational_speed_rpm}
              onChange={handleInputChange}
            />
          </div>

          <div className="form-group">
            <label>Torque [Nm]</label>
            <input
              type="number"
              name="Torque_Nm"
              value={formData.Torque_Nm}
              onChange={handleInputChange}
              step="0.1"
            />
          </div>

          <div className="form-group">
            <label>Tool Wear [min]</label>
            <input
              type="number"
              name="Tool_wear_min"
              value={formData.Tool_wear_min}
              onChange={handleInputChange}
            />
          </div>
        </div>

        <div className="button-group">
          <button className="btn-secondary" onClick={fillRandom}>
            Randomize Values
          </button>
          <button className="btn-primary" onClick={predict} disabled={loading}>
            {loading ? 'Analyzing...' : 'Predict Maintenance'}
          </button>
        </div>

        {error && (
          <div className="result failure" style={{ marginTop: '1rem' }}>
            <p>{error}</p>
          </div>
        )}

        {prediction && (
          <div className={`result ${prediction.prediction === 1 ? 'failure' : 'success'}`}>
            <h3>{prediction.prediction === 1 ? 'Maintenance Required' : 'Status: Healthy'}</h3>
            <p>
              {prediction.prediction === 1
                ? `High failure risk detected (${(prediction.failure_probability * 100).toFixed(1)}%)`
                : `Machine operating within normal parameters.`}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
