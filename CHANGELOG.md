# Changelog

## 1.5.0 (2026-01-30)

Full Changelog: [v1.4.0...v1.5.0](https://github.com/landing-ai/ade-python/compare/v1.4.0...v1.5.0)

### Features

* **client:** add custom JSON encoder for extended type support ([9c0fab4](https://github.com/landing-ai/ade-python/commit/9c0fab444b99992ac2b1da8f9af79466dd826f86))
* **client:** add support for binary request streaming ([79d0b6e](https://github.com/landing-ai/ade-python/commit/79d0b6e3756d98b0c42ec80e7582f13665b29a1a))


### Bug Fixes

* **docs:** fix mcp installation instructions for remote servers ([c164f34](https://github.com/landing-ai/ade-python/commit/c164f34af715d20af369ffcca298af055bdbf3c1))


### Chores

* **ci:** upgrade `actions/github-script` ([e6bf3e3](https://github.com/landing-ai/ade-python/commit/e6bf3e39fe120ae0e139990c5fc54dc2c1733d22))
* **internal:** update `actions/checkout` version ([b0cd317](https://github.com/landing-ai/ade-python/commit/b0cd317d0be1a2048d5dcb9586f771e4c7f026c9))

## 1.4.0 (2026-01-06)

Full Changelog: [v1.3.0...v1.4.0](https://github.com/landing-ai/ade-python/compare/v1.3.0...v1.4.0)

### Features

* **api:** api update ([7fda941](https://github.com/landing-ai/ade-python/commit/7fda941b4cbbbe00e583877013e59148c8fbb5e4))
* **files:** add support for string alternative to file upload type ([57ae8fb](https://github.com/landing-ai/ade-python/commit/57ae8fb081febb893ee7801836ccd3a725185559))


### Bug Fixes

* use async_to_httpx_files in patch method ([977143a](https://github.com/landing-ai/ade-python/commit/977143a0435f66a04314b92eab54a2145f1776ad))


### Chores

* **internal:** add `--fix` argument to lint script ([caa1bc4](https://github.com/landing-ai/ade-python/commit/caa1bc4030bc88621f898815f3c24060bbc32bf2))
* **internal:** codegen related update ([9a57490](https://github.com/landing-ai/ade-python/commit/9a57490dad6d9bb91e30cb6b476ead94aafce9d0))
* speedup initial import ([8e2a9f1](https://github.com/landing-ai/ade-python/commit/8e2a9f1b75dc7fca85fab1fb4968ccdd4da204ac))
* speedup initial import ([0b024ed](https://github.com/landing-ai/ade-python/commit/0b024ed6f2ff7fd27cb08ab1e1afeee3b3f842bf))


### Documentation

* prominently feature MCP server setup in root SDK readmes ([bb75fea](https://github.com/landing-ai/ade-python/commit/bb75fea795f468e8a215f83626ca660e1ec28d54))

## 1.3.0 (2025-12-16)

Full Changelog: [v1.2.0...v1.3.0](https://github.com/landing-ai/ade-python/compare/v1.2.0...v1.3.0)

### Features

* **api:** api update ([d6c47a8](https://github.com/landing-ai/ade-python/commit/d6c47a82dbbe6df916b9f872f99c8407b27d32cc))


### Bug Fixes

* **types:** allow pyright to infer TypedDict types within SequenceNotStr ([1070843](https://github.com/landing-ai/ade-python/commit/1070843e5b7ed4271c963c5508c580798d89c764))


### Chores

* add missing docstrings ([2d5349b](https://github.com/landing-ai/ade-python/commit/2d5349be60c9b41c7ef18fef9a2533930918a99c))
* **internal:** add missing files argument to base client ([aa24426](https://github.com/landing-ai/ade-python/commit/aa24426584864e08eb7a8980b3ae30e3771d7976))

## 1.2.0 (2025-12-04)

Full Changelog: [v1.1.1...v1.2.0](https://github.com/landing-ai/ade-python/compare/v1.1.1...v1.2.0)

### Features

* **api:** api update ([97ea48f](https://github.com/landing-ai/ade-python/commit/97ea48f456d9de814628eb8acbfa66fba843d615))


### Chores

* **docs:** use environment variables for authentication in code snippets ([d61620d](https://github.com/landing-ai/ade-python/commit/d61620daa0d544c8bfc12c9c8a663df1ed6ada42))
* update lockfile ([cf313c5](https://github.com/landing-ai/ade-python/commit/cf313c5c54a9a87ee841552411837cff8f4f3fde))

## 1.1.1 (2025-12-02)

Full Changelog: [v1.1.0...v1.1.1](https://github.com/landing-ai/ade-python/compare/v1.1.0...v1.1.1)

### Bug Fixes

* need enumeration for brackets. ([#58](https://github.com/landing-ai/ade-python/issues/58)) ([8fee225](https://github.com/landing-ai/ade-python/commit/8fee2254ca571908751bba0562456d7e843df606))

## 1.1.0 (2025-12-02)

Full Changelog: [v1.0.0...v1.1.0](https://github.com/landing-ai/ade-python/compare/v1.0.0...v1.1.0)

### Features

* **api:** manual updates ([a2c622d](https://github.com/landing-ai/ade-python/commit/a2c622db797caf02fc38e8e10ec841793b545032))


### Bug Fixes

* ensure streams are always closed ([c903098](https://github.com/landing-ai/ade-python/commit/c90309862771f208c24df5bb7a570769821d1a7a))


### Chores

* **deps:** mypy 1.18.1 has a regression, pin to 1.17 ([a768a1c](https://github.com/landing-ai/ade-python/commit/a768a1c1cf8953a48c25712e293f49bee98494dc))

## 1.0.0 (2025-11-22)

Full Changelog: [v0.21.2...v1.0.0](https://github.com/landing-ai/ade-python/compare/v0.21.2...v1.0.0)

### Chores

* add Python 3.14 classifier and testing ([962ba72](https://github.com/landing-ai/ade-python/commit/962ba7216876b06b2df1b385e9330233d57e0875))

## 0.21.2 (2025-11-12)

Full Changelog: [v0.21.1...v0.21.2](https://github.com/landing-ai/ade-python/compare/v0.21.1...v0.21.2)

### Bug Fixes

* **compat:** update signatures of `model_dump` and `model_dump_json` for Pydantic v1 ([5387db5](https://github.com/landing-ai/ade-python/commit/5387db5fc5f93493ddb424bce3f38b64b6096605))

## 0.21.1 (2025-11-11)

Full Changelog: [v0.21.0...v0.21.1](https://github.com/landing-ai/ade-python/compare/v0.21.0...v0.21.1)

### Bug Fixes

* compat with Python 3.14 ([9f26044](https://github.com/landing-ai/ade-python/commit/9f26044e9cef0625129b0440b85dcc281e4f5c3d))


### Chores

* **package:** drop Python 3.8 support ([5bfc187](https://github.com/landing-ai/ade-python/commit/5bfc187b8bc9ad0808dcdc7f1747b08880ec2ee8))

## 0.21.0 (2025-11-10)

Full Changelog: [v0.20.3...v0.21.0](https://github.com/landing-ai/ade-python/compare/v0.20.3...v0.21.0)

### Features

* **api:** api update ([6032e03](https://github.com/landing-ai/ade-python/commit/6032e03f57a4817c92668bc6433e1e17ad5fb210))

## 0.20.3 (2025-11-04)

Full Changelog: [v0.20.2...v0.20.3](https://github.com/landing-ai/ade-python/compare/v0.20.2...v0.20.3)

### Chores

* **internal:** grammar fix (it's -&gt; its) ([ab47003](https://github.com/landing-ai/ade-python/commit/ab47003b077de96f11355d29e5ff99b4f0c5f40e))

## 0.20.2 (2025-10-31)

Full Changelog: [v0.20.1...v0.20.2](https://github.com/landing-ai/ade-python/compare/v0.20.1...v0.20.2)

### Chores

* **internal/tests:** avoid race condition with implicit client cleanup ([038158c](https://github.com/landing-ai/ade-python/commit/038158c184b41e77366ff23729ebfb6c86d2db6f))

## 0.20.1 (2025-10-30)

Full Changelog: [v0.20.0...v0.20.1](https://github.com/landing-ai/ade-python/compare/v0.20.0...v0.20.1)

### Bug Fixes

* **client:** close streams without requiring full consumption ([64a689d](https://github.com/landing-ai/ade-python/commit/64a689dcb471376472acca2f9550830b6295599d))

## 0.20.0 (2025-10-29)

Full Changelog: [v0.19.0...v0.20.0](https://github.com/landing-ai/ade-python/compare/v0.19.0...v0.20.0)

### Features

* **api:** api update ([c43f1aa](https://github.com/landing-ai/ade-python/commit/c43f1aa2ee0241ade73881dfd58ccdb02a19cc8c))

## 0.19.0 (2025-10-28)

Full Changelog: [v0.18.4...v0.19.0](https://github.com/landing-ai/ade-python/compare/v0.18.4...v0.19.0)

### Features

* **api:** api update ([60362bf](https://github.com/landing-ai/ade-python/commit/60362bf6e8b8edcdad107e0807836a504d5e8964))

## 0.18.4 (2025-10-18)

Full Changelog: [v0.18.3...v0.18.4](https://github.com/landing-ai/ade-python/compare/v0.18.3...v0.18.4)

### Chores

* bump `httpx-aiohttp` version to 0.1.9 ([d442ad4](https://github.com/landing-ai/ade-python/commit/d442ad41bf47c0dab3319190992627751d7cac4a))

## 0.18.3 (2025-10-14)

Full Changelog: [v0.18.2...v0.18.3](https://github.com/landing-ai/ade-python/compare/v0.18.2...v0.18.3)

## 0.18.2 (2025-10-11)

Full Changelog: [v0.18.1...v0.18.2](https://github.com/landing-ai/ade-python/compare/v0.18.1...v0.18.2)

### Chores

* **internal:** detect missing future annotations with ruff ([033a0e0](https://github.com/landing-ai/ade-python/commit/033a0e003ecb9f6e9455385ac53d89499deea0de))

## 0.18.1 (2025-10-10)

Full Changelog: [v0.18.0...v0.18.1](https://github.com/landing-ai/ade-python/compare/v0.18.0...v0.18.1)

## 0.18.0 (2025-10-10)

Full Changelog: [v0.17.1...v0.18.0](https://github.com/landing-ai/ade-python/compare/v0.17.1...v0.18.0)

### Features

* **api:** manual updates ([5743253](https://github.com/landing-ai/ade-python/commit/57432532d8ff622e1980d892a13cad5184cc92c5))
* **api:** update via SDK Studio ([d94d74a](https://github.com/landing-ai/ade-python/commit/d94d74a9545dba1d8de7e7c616ac63b602c28c95))


### Chores

* remove custom code ([ba76a38](https://github.com/landing-ai/ade-python/commit/ba76a38dd201f37f687cf43fe6ad2605787bfd0a))

## 0.17.1 (2025-10-02)

Full Changelog: [v0.17.0...v0.17.1](https://github.com/landing-ai/ade-python/compare/v0.17.0...v0.17.1)

## 0.17.0 (2025-10-02)

Full Changelog: [v0.16.0...v0.17.0](https://github.com/landing-ai/ade-python/compare/v0.16.0...v0.17.0)

### Features

* **api:** manual updates ([a6b2c53](https://github.com/landing-ai/ade-python/commit/a6b2c5319349c82d74e56bb6c5945cd720856619))

## 0.16.0 (2025-10-02)

Full Changelog: [v0.15.1...v0.16.0](https://github.com/landing-ai/ade-python/compare/v0.15.1...v0.16.0)

### Features

* **api:** markdown commnet chaagne ([76d7de5](https://github.com/landing-ai/ade-python/commit/76d7de531313b3c268d1a5b8a32d23bc5b8682b3))

## 0.15.1 (2025-09-30)

Full Changelog: [v0.15.0...v0.15.1](https://github.com/landing-ai/ade-python/compare/v0.15.0...v0.15.1)

### Bug Fixes

* **api:** increase default timeout ([206b5d7](https://github.com/landing-ai/ade-python/commit/206b5d7567eb9f08aedbfdf1752af5cc9d1ac5c0))

## 0.15.0 (2025-09-29)

Full Changelog: [v0.14.1...v0.15.0](https://github.com/landing-ai/ade-python/compare/v0.14.1...v0.15.0)

### Features

* **api:** default models for extract ([7250c3f](https://github.com/landing-ai/ade-python/commit/7250c3f0978e5eb2d65f0535e80a5c7351d1f9f0))

## 0.14.1 (2025-09-29)

Full Changelog: [v0.14.0...v0.14.1](https://github.com/landing-ai/ade-python/compare/v0.14.0...v0.14.1)

### Bug Fixes

* add back runtime tag ([e886225](https://github.com/landing-ai/ade-python/commit/e8862252d96782b0c9dddc42042bf432b670fbd1))

## 0.14.0 (2025-09-29)

Full Changelog: [v0.13.1...v0.14.0](https://github.com/landing-ai/ade-python/compare/v0.13.1...v0.14.0)

### Features

* **api:** add extract endpoint enums ([ac88f43](https://github.com/landing-ai/ade-python/commit/ac88f431bdec9a734ed340ad00c1f9f14a1c1f49))

## 0.13.1 (2025-09-25)

Full Changelog: [v0.13.0...v0.13.1](https://github.com/landing-ai/ade-python/compare/v0.13.0...v0.13.1)

## 0.13.0 (2025-09-25)

Full Changelog: [v0.12.0...v0.13.0](https://github.com/landing-ai/ade-python/compare/v0.12.0...v0.13.0)

### Features

* **api:** update README examples to support doccument_url as local path ([f31d6ca](https://github.com/landing-ai/ade-python/commit/f31d6cabfea19aa8f152e8030a0d7d256733f7a2))

## 0.12.0 (2025-09-25)

Full Changelog: [v0.11.1...v0.12.0](https://github.com/landing-ai/ade-python/compare/v0.11.1...v0.12.0)

### Features

* document_url support local path ([#22](https://github.com/landing-ai/ade-python/issues/22)) ([5da57a5](https://github.com/landing-ai/ade-python/commit/5da57a55c0f674888a48af8d3d80b6fb5b55160c))

## 0.11.1 (2025-09-25)

Full Changelog: [v0.11.0...v0.11.1](https://github.com/landing-ai/ade-python/compare/v0.11.0...v0.11.1)

## 0.11.0 (2025-09-25)

Full Changelog: [v0.10.0...v0.11.0](https://github.com/landing-ai/ade-python/compare/v0.10.0...v0.11.0)

### Features

* **api:** change support email ([4654caf](https://github.com/landing-ai/ade-python/commit/4654caf732791296e26380ecb04b8ccae5b67551))

## 0.10.0 (2025-09-24)

Full Changelog: [v0.9.0...v0.10.0](https://github.com/landing-ai/ade-python/compare/v0.9.0...v0.10.0)

### Features

* **api:** manual updates ([13b971c](https://github.com/landing-ai/ade-python/commit/13b971c75920f9a7aadd1d576064d9fac4f3ab48))

## 0.9.0 (2025-09-24)

Full Changelog: [v0.8.1...v0.9.0](https://github.com/landing-ai/ade-python/compare/v0.8.1...v0.9.0)

### Features

* **api:** manual updates ([19e3c31](https://github.com/landing-ai/ade-python/commit/19e3c31cf6bd3f480cf6e6e928a53aa4ca259c3f))

## 0.8.1 (2025-09-24)

Full Changelog: [v0.8.0...v0.8.1](https://github.com/landing-ai/ade-python/compare/v0.8.0...v0.8.1)

## 0.8.0 (2025-09-24)

Full Changelog: [v0.7.0...v0.8.0](https://github.com/landing-ai/ade-python/compare/v0.7.0...v0.8.0)

### Features

* **api:** manual updates ([7f32e5a](https://github.com/landing-ai/ade-python/commit/7f32e5a8fa173ff0119d988466cc2edd9a1bc195))

## 0.7.0 (2025-09-23)

Full Changelog: [v0.6.1...v0.7.0](https://github.com/landing-ai/ade-python/compare/v0.6.1...v0.7.0)

### Features

* **api:** manual updates ([d2bd4c7](https://github.com/landing-ai/ade-python/commit/d2bd4c7ab65d9fcd9b898f6af80862dbe9285021))

## 0.6.1 (2025-09-23)

Full Changelog: [v0.6.0...v0.6.1](https://github.com/landing-ai/ade-python/compare/v0.6.0...v0.6.1)

## 0.6.0 (2025-09-23)

Full Changelog: [v0.5.0...v0.6.0](https://github.com/landing-ai/ade-python/compare/v0.5.0...v0.6.0)

### Features

* **api:** manual updates ([6f6ec00](https://github.com/landing-ai/ade-python/commit/6f6ec00e13f0600bf78fd909dd3154343e9ec78b))

## 0.5.0 (2025-09-22)

Full Changelog: [v0.4.0...v0.5.0](https://github.com/landing-ai/ade-python/compare/v0.4.0...v0.5.0)

### Features

* **api:** manual updates ([3f1ecbb](https://github.com/landing-ai/ade-python/commit/3f1ecbbc0665214951e5373a657d0c71187d0314))

## 0.4.0 (2025-09-22)

Full Changelog: [v0.3.0...v0.4.0](https://github.com/landing-ai/ade-python/compare/v0.3.0...v0.4.0)

### Features

* **api:** manual updates ([c4546ae](https://github.com/landing-ai/ade-python/commit/c4546aef566721f812c4f1328ef516893039087a))

## 0.3.0 (2025-09-22)

Full Changelog: [v0.2.2...v0.3.0](https://github.com/landing-ai/ade-python/compare/v0.2.2...v0.3.0)

### Features

* **api:** manual updates ([bf088ab](https://github.com/landing-ai/ade-python/commit/bf088ab5e2731d64a271608a98b86c76171bef6a))

## 0.2.2 (2025-09-22)

Full Changelog: [v0.2.1...v0.2.2](https://github.com/landing-ai/ade-python/compare/v0.2.1...v0.2.2)

### Chores

* do not install brew dependencies in ./scripts/bootstrap by default ([5848b5d](https://github.com/landing-ai/ade-python/commit/5848b5d709c7067d601ca075373fadc5dc4c337c))
* update SDK settings ([b6fafa9](https://github.com/landing-ai/ade-python/commit/b6fafa97c01d825f58b7805e58bd670bbd7b3391))

## 0.2.1 (2025-09-19)

Full Changelog: [v0.2.0...v0.2.1](https://github.com/landing-ai/ade-python/compare/v0.2.0...v0.2.1)

### Chores

* **types:** change optional parameter type from NotGiven to Omit ([29a0a2d](https://github.com/landing-ai/ade-python/commit/29a0a2de368b135025a8379e26634f4dc5d6a1e8))

## 0.2.0 (2025-09-18)

Full Changelog: [v0.1.0...v0.2.0](https://github.com/landing-ai/ade-python/compare/v0.1.0...v0.2.0)

### Features

* **api:** support environments ([e9b604e](https://github.com/landing-ai/ade-python/commit/e9b604e76d03a9e630c8567d3f014032ca186376))

## 0.1.0 (2025-09-18)

Full Changelog: [v0.0.1...v0.1.0](https://github.com/landing-ai/ade-python/compare/v0.0.1...v0.1.0)

### Features

* **api:** manual updates ([eb76a32](https://github.com/landing-ai/ade-python/commit/eb76a3275704d50396d00fd8ac79c2537ce251fc))


### Chores

* configure new SDK language ([9761e2b](https://github.com/landing-ai/ade-python/commit/9761e2bed207087deba958e693fd381eb5599a67))
* update SDK settings ([b46e740](https://github.com/landing-ai/ade-python/commit/b46e74012a27713aaa82f99bd11e527c92e912f4))
* update SDK settings ([982e228](https://github.com/landing-ai/ade-python/commit/982e2280ef59753578cfc5c4272fca2f90c2083a))
