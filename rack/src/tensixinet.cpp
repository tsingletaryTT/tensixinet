// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// tensixinet Rack: an embedded CPython host for the real tensixinet synthesizer.
#include <rack.hpp>
#include <Python.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <cstring>

using namespace rack;

extern Model *modelTensixinet;

namespace {

// Paths are discovered at startup from environment or defaults, never hard-wired to a home dir alone.
std::string envOr(const char *name, const char *fallback) {
	const char *v = std::getenv(name);
	return v && *v ? std::string(v) : std::string(fallback);
}

static const int N_INPUT_PORTS = 24;
static const char *INPUT_PORT_NAMES[N_INPUT_PORTS] = {
	"temperature", "power", "aiclk", "voltage", "gddr0", "gddr1", "gddr2", "gddr3",
	"gddr4", "gddr5", "gddr6", "gddr7", "ddr_status", "eth_links", "tensix", "pressure",
	"heartbeat", "temp_rate", "power_rate", "busy", "corr_errs", "uncorr_errs", "thermal_trips", "style"
};

struct Engine {
	std::mutex lock;
	bool ready = false;
	std::string error;
	std::string banner;

	void start() {
		std::lock_guard<std::mutex> guard(lock);
		if (ready || !error.empty()) return;
		std::string root = envOr("TENSIXINET_ROOT", "/home/ttuser/code/tensixinet");
		std::string site = envOr("TENSIXINET_SITE", "/home/ttuser/.tenstorrent-venv/lib/python3.12/site-packages");
		std::string dynload = envOr("TENSIXINET_DYNLOAD", "/usr/lib/python3.12/lib-dynload");
		if (!Py_IsInitialized()) {
			PyStatus status;
			PyConfig config;
			PyConfig_InitIsolatedConfig(&config);
			PyConfig_SetString(&config, &config.program_name, L"Rack");
			PyConfig_SetString(&config, &config.home, L"/usr");
			config.site_import = 0;
			config.user_site_directory = 0;
			config.write_bytecode = 0;
			config.parse_argv = 0;
			status = Py_InitializeFromConfig(&config);
			PyConfig_Clear(&config);
			if (PyStatus_Exception(status)) {
				error = std::string("Python init failed: ") + (status.err_msg ? status.err_msg : "unknown");
				return;
			}
		}
		PyGILState_STATE gil = PyGILState_Ensure();
		PyObject *sysPath = PySys_GetObject("path");  // borrowed
		for (const char *p : {root.c_str(), site.c_str(), dynload.c_str()})
			PyList_Append(sysPath, PyUnicode_FromString(p));
		const char *code =
			"import numpy as np\n"
			"from tensixinet.rack.cv import RackBridge\n"
			"_tsx = RackBridge(chips=4, sample_rate=44100, source='mock')\n"
			"print('tensixinet Rack: Python bridge ready,', len(_tsx.readings), 'mock chips')";
		if (PyRun_SimpleString(code) != 0) {
			error = "Python bridge failed. Set TENSIXINET_ROOT and TENSIXINET_SITE to your Python environment.";
			banner = error;
		} else {
			ready = true;
			banner = "tensixinet Rack: embedded CPython, 4 mock chips";
		}
		PyGILState_Release(gil);
	}

	void setInputs(const std::vector<float> &values, int source) {
		if (!ready) return;
		PyGILState_STATE gil = PyGILState_Ensure();
		PyObject *d = PyDict_New();
		for (int i = 0; i < N_INPUT_PORTS && i < (int)values.size(); i++)
			PyDict_SetItemString(d, INPUT_PORT_NAMES[i], PyFloat_FromDouble(values[i]));
		PyObject *m = PyImport_AddModule("__main__");  // borrowed
		PyObject *dict = PyModule_GetDict(m);  // borrowed
		PyObject *bridge = PyDict_GetItemString(dict, "_tsx");  // borrowed
		if (bridge) {
			PyObject_CallMethod(bridge, "set_params", "Oi", d, source);
		}
		Py_DECREF(d);
		PyGILState_Release(gil);
	}


	// Pull the outgoing CV streams for the segment that was just rendered.
	bool streams(std::vector<std::vector<float>> &dst, std::string &err) {
		if (!ready) return false;
		PyGILState_STATE gil = PyGILState_Ensure();
		PyObject *m = PyImport_AddModule("__main__");
		PyObject *dict = PyModule_GetDict(m);
		PyObject *bridge = PyDict_GetItemString(dict, "_tsx");
		bool ok = false;
		if (bridge) {
			PyObject *r = PyObject_CallMethod(bridge, "cv_streams", nullptr);
			if (r && PySequence_Check(r)) {
				Py_ssize_t n = PySequence_Size(r);
				dst.clear();
				for (Py_ssize_t i = 0; i < n; i++) {
					PyObject *row = PySequence_GetItem(r, i);
					Py_buffer view;
					if (row && PyObject_GetBuffer(row, &view, PyBUF_SIMPLE) == 0) {
						const float *p = (const float *)view.buf;
						dst.emplace_back(p, p + view.len / (Py_ssize_t)sizeof(float));
						PyBuffer_Release(&view);
					}
					Py_XDECREF(row);
				}
				ok = true;
			}
			Py_XDECREF(r);
			if (!ok && PyErr_Occurred()) PyErr_Print();
		}
		PyGILState_Release(gil);
		if (!ok) err = "CV streams failed";
		return ok;
	}

	// Render one short segment from Python. The frame is an ndarray; copy it through the buffer protocol.
	bool render(float seconds, int segment, std::vector<float> &out, std::string &err) {
		if (!ready) { err = error.empty() ? "Python bridge not ready" : error; return false; }
		PyGILState_STATE gil = PyGILState_Ensure();
		PyObject *m = PyImport_AddModule("__main__");
		PyObject *dict = PyModule_GetDict(m);
		PyObject *bridge = PyDict_GetItemString(dict, "_tsx");
		bool ok = false;
		if (bridge) {
			PyObject *r = PyObject_CallMethod(bridge, "render", "fi", (double)seconds, segment);
			if (r) {
				PyObject *frame = PyTuple_GetItem(r, 0);
				if (frame) {
					Py_buffer view;
					if (PyObject_GetBuffer(frame, &view, PyBUF_SIMPLE) == 0) {
						const float *p = (const float *)view.buf;
						Py_ssize_t n = view.len / (Py_ssize_t)sizeof(float);
						out.assign(p, p + n);
						PyBuffer_Release(&view);
						ok = true;
					}
				}
				Py_DECREF(r);
			}
			if (!ok) {
				if (PyErr_Occurred()) PyErr_Print();
				err = "render failed; see Rack log";
			}
		}
		PyGILState_Release(gil);
		return ok;
	}
};

static Engine &tsxEngine() {
	static Engine e;
	return e;
}

const int HP = 30;
const int PORTS = 24;
const std::vector<std::string> INPUT_NAMES = {
	"temperature", "power", "aiclk", "voltage", "gddr0", "gddr1", "gddr2", "gddr3",
	"gddr4", "gddr5", "gddr6", "gddr7", "ddr_status", "eth_links", "tensix", "pressure",
	"heartbeat", "temp_rate", "power_rate", "busy", "corr_errs", "uncorr_errs", "thermal_trips", "style"
};

struct Tensixinet : Module {
	enum ParamIds {
		SOURCE_KNOB, SEGMENT_KNOB, DURATION_KNOB, STYLE_KNOB, CHAOS_KNOB, FLOOR_KNOB,
		INPUT_ATTEN, OUT_ATTEN = INPUT_ATTEN + N_INPUT_PORTS, NUM_PARAMS = OUT_ATTEN + 2
	};
	enum InputIds { CV_INPUT, NUM_INPUTS = CV_INPUT + N_INPUT_PORTS };
	enum OutputIds { AUDIO_OUT, CTRL_OUT = AUDIO_OUT + 2, NUM_OUTPUTS = CTRL_OUT + 28 };
	enum LightIds { STATUS_LIGHT, NUM_LIGHTS = STATUS_LIGHT + 1 };

	std::atomic<int> pendingSegment{0};
	std::vector<float> pendingCV;
	std::mutex pendingLock;
	std::vector<float> frame;
	std::vector<std::vector<float>> streams;
	size_t playhead = 0;
	std::string status;
	std::thread worker;
	std::atomic<bool> running{true};

	Tensixinet() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(SOURCE_KNOB, 0.f, 3.f, 0.f, "Source", "", 0.f, 1.f);
		configParam(SEGMENT_KNOB, 0.f, 1.f, 0.f, "Segment", "", 0.f, 1.f);
		configParam(DURATION_KNOB, 0.1f, 2.f, 0.25f, "Segment length", "s");
		configParam(STYLE_KNOB, 0.f, 4.f, 0.f, "Style", "", 0.f, 1.f);
		configParam(CHAOS_KNOB, 0.f, 1.f, 0.5f, "Chaos", "%", 0.f, 100.f);
		configParam(FLOOR_KNOB, 0.f, 1.f, 0.2f, "Floor", "%", 0.f, 100.f);
		for (int i = 0; i < N_INPUT_PORTS; i++) {
			configParam(INPUT_ATTEN + i, 0.f, 1.f, 1.f, INPUT_PORT_NAMES[i], "%", 0.f, 100.f);
			configInput(CV_INPUT + i, INPUT_PORT_NAMES[i]);
		}
		for (int i = 0; i < 2; i++) configOutput(AUDIO_OUT + i, i == 0 ? "Audio L" : "Audio R");
		configOutput(CTRL_OUT + 0, "CPU temperature"); configOutput(CTRL_OUT + 1, "CPU power");
		configOutput(CTRL_OUT + 2, "CPU clock"); configOutput(CTRL_OUT + 3, "CPU pressure");
		configOutput(CTRL_OUT + 4, "Chip temperature"); configOutput(CTRL_OUT + 5, "Chip power");
		configOutput(CTRL_OUT + 6, "Chip clock"); configOutput(CTRL_OUT + 7, "Chip pressure");
		const char *cvNames[28] = {"CPU T", "CPU P", "CPU CLK", "CPU PRS", "Chip T", "Chip P", "Chip CLK", "Chip PRS",
			"Other T", "Other P", "Other CLK", "Other PRS", "Gong", "Lead V/oct", "Lead gate", "Lead velocity",
			"Quantus V/oct", "Quantus gate", "Quantus velocity", "Gong", "Snare", "Roll clock",
			"Roll 0", "Roll 1", "Roll 2", "Roll 3", "Roll 4", "Roll 5"};
		for (int i = 0; i < 28; i++) configOutput(CTRL_OUT + i, cvNames[i]);
		pendingCV.assign(INPUT_NAMES.size(), 0.f);
		tsxEngine().start();
		status = tsxEngine().error.empty() ? tsxEngine().banner : tsxEngine().error;
		worker = std::thread([this] {
			while (running.load()) {
				int seg;
				{
					std::lock_guard<std::mutex> g(pendingLock);
					seg = pendingSegment.load();
				}
				std::vector<float> next;
				std::string err;
				float seconds = params[DURATION_KNOB].getValue();
				if (tsxEngine().render(seconds, seg, next, err)) {
					frame = std::move(next);
					std::vector<std::vector<float>> cv;
					std::string cvErr;
					if (tsxEngine().streams(cv, cvErr)) streams = std::move(cv);
					playhead = 0;
					status = "rendering";
				} else status = err;
				{
					FILE *f = fopen("/tmp/tsx_rack_debug.txt", "a");
					if (f) { fprintf(f, "render frame=%zu status=%s\n", frame.size(), status.c_str()); fclose(f); }
				}
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}
		});
	}

	~Tensixinet() {
		running.store(false);
		if (worker.joinable()) worker.join();
	}

	void process(const ProcessArgs &args) override {
		// Poll the patch cables into the Python bridge twice per audio block.
		std::vector<float> voltages(INPUT_NAMES.size());
		bool changed = false;
		for (int i = 0; i < N_INPUT_PORTS; i++) {
			float v = inputs[CV_INPUT + i].getVoltage() * params[INPUT_ATTEN + i].getValue();
			voltages[i] = v;
			if (v != pendingCV[i]) changed = true;
		}
		if (changed) {
			tsxEngine().setInputs(voltages, (int)params[SOURCE_KNOB].getValue());
			{
				std::lock_guard<std::mutex> g(pendingLock);
				pendingCV = voltages;
			}
		}
		int seg = (int)params[SEGMENT_KNOB].getValue();
		if (seg != pendingSegment.load()) pendingSegment.store(seg);
		lights[STATUS_LIGHT].setSmoothBrightness(tsxEngine().ready ? 1.f : 0.f, args.sampleTime);
		if (!tsxEngine().ready || frame.empty()) return;
		if (playhead + 2 <= frame.size()) {
			outputs[AUDIO_OUT + 0].setVoltage(frame[playhead++] * 5.f);
			outputs[AUDIO_OUT + 1].setVoltage(frame[playhead++] * 5.f);
		}
		// Every outgoing jack carries the event stream from the same Score the audio came from.
		for (size_t i = 0; i < streams.size() && i < (size_t)(NUM_OUTPUTS - CTRL_OUT); i++)
			if (playhead > 0 && playhead - 1 < streams[i].size())
				outputs[CTRL_OUT + i].setVoltage(streams[i][playhead - 1]);
	}
};

struct TensixinetWidget : ModuleWidget {
	TensixinetWidget(Tensixinet *module) {
		setModule(module);
		setPanel(createPanel(asset::plugin(rack::plugin::getPlugin("tensixinet"), "res/tensixinet.svg")));
		// The panel is drawn in res/tensixinet.svg; coordinates track its 30 HP grid.
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		const float xL = 7.f, xR = 88.f, xC = 155.f;
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(xL + 8, 18)), module, Tensixinet::SOURCE_KNOB));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(xL + 8, 36)), module, Tensixinet::SEGMENT_KNOB));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(xL + 8, 54)), module, Tensixinet::DURATION_KNOB));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(xL + 8, 72)), module, Tensixinet::STYLE_KNOB));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(xL + 8, 90)), module, Tensixinet::CHAOS_KNOB));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(xL + 8, 108)), module, Tensixinet::FLOOR_KNOB));
		addChild(createLightCentered<MediumLight<GreenLight>>(mm2px(Vec(xL + 8, 122)), module, Tensixinet::STATUS_LIGHT));
		for (int i = 0; i < N_INPUT_PORTS; i++) {
			float y = 12.f + i * 5.3f;
			addInput(createInputCentered<PJ301MPort>(mm2px(Vec(xR, y)), module, Tensixinet::CV_INPUT + i));
			addParam(createParamCentered<Trimpot>(mm2px(Vec(xR + 9, y)), module, Tensixinet::INPUT_ATTEN + i));
		}
		for (int i = 0; i < 2; i++)
			addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(xC + 8, 12.f + i * 8.f)), module, Tensixinet::AUDIO_OUT + i));
		for (int i = 0; i < 28; i++)
			addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(xC + (i % 4) * 7, 30.f + (i / 4) * 8)), module, Tensixinet::CTRL_OUT + i));
	}
};

}  // namespace

Model *modelTensixinet = createModel<Tensixinet, TensixinetWidget>("Tensixinet");

extern "C" void init(rack::plugin::Plugin *plugin) {
	FILE *f = fopen("/tmp/tsx_rack_debug.txt", "a");
	if (f) { fprintf(f, "plugin init called model=%p\n", (void*)modelTensixinet); fclose(f); }
	plugin->addModel(modelTensixinet);
	tsxEngine().start();
}
