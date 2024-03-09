import {useEffect} from 'react'
import logo from './logo.svg';
import './App.css';
import {jsbgl} from '@naftalimurgor/jsbgl-react-native'

function App() {
  useEffect(() => {
    initJsbglModule()

    async function initJsbglModule() {
      await jsbgl.asyncInit()
      console.log(window);
    }

  }, [])

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
