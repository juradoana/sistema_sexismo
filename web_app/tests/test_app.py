# test_app.py
# Pruebas de integración para el servidor Flask (app.py)
# Ejecutar con: python -m pytest tests/test_app.py -v
#
# Estas pruebas usan mocking para simular los modelos de IA,
# permitiendo ejecutar las pruebas sin necesidad de GPU ni conexión al servidor LLM.

import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock

# Asegurar que el directorio padre está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_classifier():
    """Simula el clasificador RoBERTa-BNE."""
    mock = MagicMock()
    mock.predict.return_value = (True, 0.91)  # sexista, confianza 91%
    return mock


@pytest.fixture
def mock_llm_api():
    """Simula la API del LLM Gemma."""
    mock = MagicMock()
    mock.generate_response.return_value = json.dumps({
        "explicacion": "Esta frase es sexista porque generaliza.",
        "contranarrativa": "La habilidad no depende del género."
    })
    return mock


@pytest.fixture
def client(mock_classifier, mock_llm_api):
    """Crea un cliente de pruebas de Flask con los modelos simulados."""
    # Simular la carga de recursos para evitar cargar modelos reales
    with patch('app.load_resources'):
        import app as flask_app
        flask_app.classifier = mock_classifier
        flask_app.llm_api = mock_llm_api
        flask_app.app.config['TESTING'] = True

        with flask_app.app.test_client() as client:
            yield client


class TestAnalyzeEndpointSuccess:
    """PI-01, PI-04: Pruebas de peticiones válidas al endpoint /api/analyze."""

    def test_peticion_valida_retorna_200(self, client):
        """PI-01: Una petición válida debe retornar HTTP 200."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "texto de prueba", "strategy": "0-shot"}),
            content_type='application/json'
        )
        assert response.status_code == 200

    def test_respuesta_contiene_campos_esperados(self, client):
        """PI-01: La respuesta debe contener is_sexist, confidence, text y explanation."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "texto de prueba", "strategy": "0-shot"}),
            content_type='application/json'
        )
        data = json.loads(response.data)

        assert "is_sexist" in data
        assert "confidence" in data
        assert "text" in data
        assert "explanation" in data

    def test_respuesta_sexista_incluye_contranarrativa(self, client):
        """PI-01: Si el texto es sexista, la respuesta debe incluir counter_narrative."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "texto sexista", "strategy": "0-shot"}),
            content_type='application/json'
        )
        data = json.loads(response.data)

        assert data["is_sexist"] is True
        assert "counter_narrative" in data

    def test_respuesta_json_parseable(self, client):
        """PI-04: La respuesta debe ser JSON válido y parseable."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "texto", "strategy": "few-shot"}),
            content_type='application/json'
        )

        # No debe lanzar excepción al parsear
        data = json.loads(response.data)
        assert isinstance(data, dict)

    def test_confianza_es_float_entre_0_y_1(self, client):
        """La confianza debe ser un float entre 0 y 1."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "texto", "strategy": "0-shot"}),
            content_type='application/json'
        )
        data = json.loads(response.data)

        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0


class TestAnalyzeEndpointEmptyText:
    """PI-02: Pruebas de peticiones con texto vacío."""

    def test_texto_vacio_retorna_400(self, client):
        """PI-02: Texto vacío debe retornar HTTP 400."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "", "strategy": "0-shot"}),
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_texto_solo_espacios_retorna_400(self, client):
        """PI-02: Texto con solo espacios debe retornar HTTP 400."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "   ", "strategy": "0-shot"}),
            content_type='application/json'
        )
        assert response.status_code == 400

    def test_texto_vacio_retorna_mensaje_error(self, client):
        """PI-02: La respuesta de error debe incluir un mensaje descriptivo."""
        response = client.post('/api/analyze',
            data=json.dumps({"text": "", "strategy": "0-shot"}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        assert "error" in data


class TestAnalyzeEndpointNoClassifier:
    """PI-03: Pruebas cuando el clasificador no está cargado."""

    def test_sin_clasificador_retorna_500(self):
        """PI-03: Sin clasificador cargado debe retornar HTTP 500."""
        with patch('app.load_resources'):
            import app as flask_app
            flask_app.classifier = None  # Simular que el clasificador no se cargó
            flask_app.llm_api = MagicMock()
            flask_app.app.config['TESTING'] = True

            with flask_app.app.test_client() as client:
                response = client.post('/api/analyze',
                    data=json.dumps({"text": "texto de prueba", "strategy": "0-shot"}),
                    content_type='application/json'
                )
                assert response.status_code == 500

    def test_sin_clasificador_no_colapsa(self):
        """PI-03: Sin clasificador, el servidor no debe colapsar."""
        with patch('app.load_resources'):
            import app as flask_app
            flask_app.classifier = None
            flask_app.llm_api = MagicMock()
            flask_app.app.config['TESTING'] = True

            with flask_app.app.test_client() as client:
                response = client.post('/api/analyze',
                    data=json.dumps({"text": "texto", "strategy": "0-shot"}),
                    content_type='application/json'
                )
                # El servidor debe responder (no colapsar)
                assert response.status_code in [400, 500]
                data = json.loads(response.data)
                assert "error" in data


class TestIndexRoute:
    """Pruebas de la ruta principal."""

    def test_index_retorna_200(self, client):
        """La ruta raíz debe retornar HTTP 200."""
        response = client.get('/')
        assert response.status_code == 200

    def test_index_retorna_html(self, client):
        """La ruta raíz debe retornar HTML."""
        response = client.get('/')
        assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data
