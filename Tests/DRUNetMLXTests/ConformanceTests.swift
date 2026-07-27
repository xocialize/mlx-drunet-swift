// ConformanceTests.swift — DRUNet through the engine's offline gates (no MLX kernels run).
//
// The variant-specific tests matter more here than in the sibling packages, because this is the
// first multi-variant port where a variant changes the *key set* rather than just the weights:
// the denoising checkpoint is BiasFree and has no bias vectors at all.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
import DRUNetMLXCore
@testable import MLXDRUNet

final class ConformanceTests: XCTestCase {

    // MARK: - MAT

    func testMATGate() {
        let report = MaterializationConformance.check(freshConfiguration: DRUNetConfiguration())
        XCTAssertTrue(report.passed, report.summary)
    }

    func testWeightSourcesDeclaredForEveryVariant() {
        var repos = Set<String>()
        for variant in DRUNetVariant.allCases {
            let sources = DRUNetConfiguration(variant: variant).weightSources
            XCTAssertEqual(sources.count, 1, "\(variant)")
            XCTAssertEqual(sources[0].repo, variant.repo)
            XCTAssertEqual(sources[0].matching, ["model.safetensors"], "\(variant)")
            repos.insert(sources[0].repo)
        }
        // Each variant is a different product and must resolve to its own repo — a shared repo
        // would silently serve one checkpoint for all three.
        XCTAssertEqual(repos.count, DRUNetVariant.allCases.count)
    }

    func testExplicitWeightsURLSuppressesMaterialization() throws {
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("drunet-\(UUID().uuidString).safetensors")
        try Data([0x00]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }
        XCTAssertTrue(DRUNetConfiguration(weightsURL: tmp).missingWeightSources(storeRoot: nil).isEmpty)
        XCTAssertEqual(
            DRUNetConfiguration(weightsURL: tmp.appendingPathExtension("nope"))
                .missingWeightSources(storeRoot: nil).count, 1)
    }

    // MARK: - CAN

    func testCANGatePreCancelledRun() async {
        let package = DRUNetRestorePackage(configuration: DRUNetConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: ImageRestoreRequest(image: Image(format: .png, data: Data())))
        XCTAssertTrue(report.passed, report.summary)
    }

    func testCANCadenceDeclaration() {
        let manifest = DRUNetRestorePackage.manifest
        XCTAssertTrue(CancellationConformance.longRunImplied(by: manifest),
                      "4.5 GB declared activation should imply long runs")
        // run() has a REAL iterative seam — the tile loop — and checkpoints once per tile via the
        // core's onTile hook, reporting RunProgress on the same unit.
        let report = CancellationConformance.checkCadence(
            manifest: manifest,
            posture: .cadence([
                .init(phase: .postprocess, unit: .chunk, reportsRunProgress: true),
                .init(phase: .encode, unit: .frame),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Manifest

    func testManifestSurfacesAndLicence() {
        let m = DRUNetRestorePackage.manifest
        XCTAssertEqual(m.capabilities, [.imageRestore], "no new capability — same request shape")
        XCTAssertEqual(m.surfaces.count, 1)
        XCTAssertEqual(m.surfaces[0].name, "drunet-denoise")
        XCTAssertEqual(m.license.weightLicense, .mit)
        XCTAssertEqual(m.license.portCodeLicense, .mit)
        XCTAssertEqual(m.provenance.sourceRepo, "cszn/DPIR")
    }

    func testFootprintIsSplitAndFlatInResolution() {
        guard let fp = DRUNetRestorePackage.manifest.requirements.footprints
            .first(where: { $0.quant == .fp32 }) else { return XCTFail("no fp32 footprint") }
        // 32.6 M params @ fp32 = 130.6 MB; the floor must cover it without absorbing the activation.
        XCTAssertGreaterThan(fp.residentBytes, 130_600_000)
        XCTAssertLessThan(fp.residentBytes, 500_000_000)
        XCTAssertGreaterThan(fp.peakActivationBytes, fp.residentBytes)
        // DRUNet runs whole frames (no tiling), but a plain conv U-Net has no channel blow-up, so
        // the activation should stay in the same band as the tiled siblings rather than exploding.
        XCTAssertLessThan(fp.peakActivationBytes, 10_000_000_000)
    }

    func testQuantConfiguredMatchesADeclaredFootprint() {
        let declared = Set(DRUNetRestorePackage.manifest.requirements.footprints.map(\.quant))
        for v in DRUNetVariant.allCases {
            XCTAssertTrue(declared.contains(DRUNetConfiguration(variant: v).quant), "\(v)")
        }
    }

    // MARK: - The strength dial (contract 1.30.0)

    /// The whole reason contract 1.30.0 grew `strength`. If the descriptor stops advertising it,
    /// a planner silently loses the only continuous denoise control in the fleet.
    func testDescriptorAdvertisesStrength() {
        let d = DRUNetRestorePackage.manifest.surfaces[0]
        let names = d.parameters.map(\.name)
        XCTAssertTrue(names.contains("image"))
        XCTAssertTrue(names.contains("strength"),
                      "DRUNet is the backer that HAS a dial: \(names)")
        XCTAssertEqual(d.parameters.first { $0.name == "strength" }?.required, false)
    }

    /// strength → σ is the mapping the whole feature rides on. Pin the endpoints and the default so
    /// a change is deliberate.
    func testStrengthMapsOntoSigma() {
        let c = DRUNetConfiguration()
        XCTAssertEqual(c.sigma(for: 0), 0, accuracy: 1e-6, "0 must mean no denoise")
        XCTAssertEqual(c.sigma(for: 1), c.maxSigma, accuracy: 1e-6)
        XCTAssertEqual(c.sigma(for: 0.5), c.maxSigma / 2, accuracy: 1e-6)
        // Out-of-range input is clamped, not extrapolated into territory the checkpoint was never
        // evaluated at.
        XCTAssertEqual(c.sigma(for: -5), 0, accuracy: 1e-6)
        XCTAssertEqual(c.sigma(for: 99), c.maxSigma, accuracy: 1e-6)
        // nil = the caller expressed no preference → the paper's mildest level, not the max.
        XCTAssertEqual(c.sigma(for: nil), c.defaultSigma, accuracy: 1e-6)
        XCTAssertLessThan(c.defaultSigma, c.maxSigma,
                          "the no-preference default must be conservative")
    }

    /// maxSigma is capped at the paper's top reported level rather than the 0…255 nominal range —
    /// pushing past what the authors characterised would be extrapolation.
    func testMaxSigmaStaysWithinCharacterisedRange() {
        XCTAssertEqual(DRUNetConfiguration().maxSigma, 50, accuracy: 1e-6)
        XCTAssertEqual(DRUNetConfiguration().defaultSigma, 15, accuracy: 1e-6)
    }

    func testVariantsResolveToDistinctRepos() {
        XCTAssertEqual(Set(DRUNetVariant.allCases.map(\.repo)).count, DRUNetVariant.allCases.count)
    }
}
